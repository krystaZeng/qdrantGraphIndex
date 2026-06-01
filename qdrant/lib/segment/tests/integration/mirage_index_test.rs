use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use common::types::ScoredPointOffset;
use rand::SeedableRng;
use rand::prelude::StdRng;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::SegmentEntry;
use segment::fixtures::payload_fixtures::random_vector;
use segment::id_tracker::IdTracker;
use segment::index::mirage_index::{MIRAGE_INDEX_CONFIG_FILE, MirageIndex, MirageIndexOpenArgs};
use segment::index::{VectorIndex, VectorIndexEnum};
use segment::segment::Segment;
use segment::segment_constructor::VectorIndexBuildArgs;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Distance, ExtendedPointId, HnswGlobalConfig, Indexes, MirageConfig, SearchParams,
    SegmentConfig, SeqNumberType, VectorDataConfig, VectorStorageType,
};
use tempfile::Builder;
use uuid::Uuid;

const DIM: usize = 128;
const NUM_VECTORS: u64 = 2_000;
const TOP: usize = 100;
const ATTEMPTS: usize = 50;
const SEARCH_EF: usize = 256;
const MIN_AVG_RECALL: f64 = 0.70;

fn mirage_config() -> MirageConfig {
    MirageConfig {
        m: 16,
        ef_construct: 1024,
        full_scan_threshold: 1,
        max_indexing_threads: 1,
        on_disk: Some(false),
        payload_m: None,
        ..Default::default()
    }
}

fn mirage_segment_config() -> SegmentConfig {
    SegmentConfig {
        vector_data: HashMap::from([(
            DEFAULT_VECTOR_NAME.to_owned(),
            VectorDataConfig {
                size: DIM,
                distance: Distance::Euclid,
                storage_type: VectorStorageType::InRamChunkedMmap,
                index: Indexes::Mirage(mirage_config()),
                quantization_config: None,
                multivector_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
    }
}

fn build_plain_segment(prefix: &str, seed: u64) -> Segment {
    let dir = Builder::new().prefix(prefix).tempdir().unwrap();
    let hw_counter = HardwareCounterCell::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut segment = build_simple_segment(dir.path(), DIM, Distance::Euclid).unwrap();

    for n in 0..NUM_VECTORS {
        let vector = random_vector(&mut rng, DIM);
        segment
            .upsert_point(
                n as SeqNumberType,
                n.into(),
                only_default_vector(&vector),
                &hw_counter,
            )
            .unwrap();
    }

    segment
}

fn result_external_ids(
    segment: &Segment,
    result: &[ScoredPointOffset],
) -> HashSet<ExtendedPointId> {
    let id_tracker = segment.id_tracker.borrow();
    result
        .iter()
        .map(|point| id_tracker.external_id(point.idx).unwrap())
        .collect()
}

fn average_recall_against_plain(
    exact_segment: &Segment,
    approximate_segment: &Segment,
    seed: u64,
    mut approximate_search: impl FnMut(&QueryVector) -> Vec<ScoredPointOffset>,
) -> f64 {
    let mut query_rng = StdRng::seed_from_u64(seed);
    let mut recall_sum = 0.0;

    for _ in 0..ATTEMPTS {
        let query = random_vector(&mut query_rng, DIM).into();

        let exact = exact_segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&query], None, TOP, None, &Default::default())
            .unwrap()
            .remove(0);

        let approx = approximate_search(&query);

        let exact_ids = result_external_ids(exact_segment, &exact);
        let approx_ids = result_external_ids(approximate_segment, &approx);
        let overlap = exact_ids.intersection(&approx_ids).count();
        recall_sum += overlap as f64 / TOP as f64;
    }

    recall_sum / ATTEMPTS as f64
}

#[test]
fn mirage_index_direct_build_open_search_recall() {
    let stopped = AtomicBool::new(false);
    let segment = build_plain_segment("mirage_direct_segment", 42);
    let mirage_dir = Builder::new()
        .prefix("mirage_direct_index")
        .tempdir()
        .unwrap();
    let mut build_rng = StdRng::seed_from_u64(7);
    let config = mirage_config();

    let mirage_index = MirageIndex::build(
        MirageIndexOpenArgs {
            path: mirage_dir.path(),
            id_tracker: segment.id_tracker.clone(),
            vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_storage
                .clone(),
            quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
                .quantized_vectors
                .clone(),
            payload_index: segment.payload_index.clone(),
            mirage_config: config,
        },
        VectorIndexBuildArgs {
            permit: Arc::new(ResourcePermit::dummy(1)),
            old_indices: &[],
            gpu_device: None,
            rng: &mut build_rng,
            stopped: &stopped,
            hnsw_global_config: &HnswGlobalConfig::default(),
            feature_flags: FeatureFlags::default(),
            progress: ProgressTracker::new_for_test(),
        },
    )
    .unwrap();

    assert_eq!(mirage_index.indexed_vector_count(), NUM_VECTORS as usize);
    assert!(mirage_dir.path().join(MIRAGE_INDEX_CONFIG_FILE).exists());
    drop(mirage_index);

    let mirage_index = MirageIndex::open(MirageIndexOpenArgs {
        path: mirage_dir.path(),
        id_tracker: segment.id_tracker.clone(),
        vector_storage: segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .clone(),
        quantized_vectors: segment.vector_data[DEFAULT_VECTOR_NAME]
            .quantized_vectors
            .clone(),
        payload_index: segment.payload_index.clone(),
        mirage_config: config,
    })
    .unwrap();

    let avg_recall = average_recall_against_plain(&segment, &segment, 100, |query| {
        mirage_index
            .search(
                &[query],
                None,
                TOP,
                Some(&SearchParams {
                    hnsw_ef: Some(SEARCH_EF),
                    ..Default::default()
                }),
                &Default::default(),
            )
            .unwrap()
            .remove(0)
    });
    eprintln!("direct Mirage recall@{TOP}: {avg_recall:.3}");

    assert!(
        avg_recall >= MIN_AVG_RECALL,
        "Mirage recall@{TOP} too low: {avg_recall:.3}, expected >= {MIN_AVG_RECALL:.3}",
    );
}

#[test]
fn mirage_segment_constructor_builds_vector_index_enum_mirage() {
    let stopped = AtomicBool::new(false);
    let hw_counter = HardwareCounterCell::new();
    let source_segment = build_plain_segment("mirage_constructor_source", 42);
    let output_dir = Builder::new()
        .prefix("mirage_constructor_output")
        .tempdir()
        .unwrap();
    let temp_dir = Builder::new()
        .prefix("mirage_constructor_temp")
        .tempdir()
        .unwrap();

    let mut builder = SegmentBuilder::new(
        temp_dir.path(),
        &mirage_segment_config(),
        &HnswGlobalConfig::default(),
    )
    .unwrap();
    builder.update(&[&source_segment], &stopped).unwrap();

    let mut build_rng = StdRng::seed_from_u64(7);
    let built_segment = builder
        .build(
            output_dir.path(),
            Uuid::new_v4(),
            None,
            ResourcePermit::dummy(1),
            &stopped,
            &mut build_rng,
            &hw_counter,
            ProgressTracker::new_for_test(),
        )
        .unwrap();

    let vector_index = built_segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();

    match &*vector_index {
        VectorIndexEnum::Mirage(index) => {
            assert_eq!(index.indexed_vector_count(), NUM_VECTORS as usize);
        }
        other => panic!("expected VectorIndexEnum::Mirage, got {other:?}"),
    }

    let avg_recall = average_recall_against_plain(&source_segment, &built_segment, 100, |query| {
        vector_index
            .search(
                &[query],
                None,
                TOP,
                Some(&SearchParams {
                    hnsw_ef: Some(SEARCH_EF),
                    ..Default::default()
                }),
                &Default::default(),
            )
            .unwrap()
            .remove(0)
    });
    eprintln!("constructor Mirage recall@{TOP}: {avg_recall:.3}");

    assert!(
        avg_recall >= MIN_AVG_RECALL,
        "Mirage constructor recall@{TOP} too low: {avg_recall:.3}, expected >= {MIN_AVG_RECALL:.3}",
    );
}
