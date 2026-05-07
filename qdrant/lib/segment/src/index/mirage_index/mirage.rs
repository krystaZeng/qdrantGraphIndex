//! MIRAGE vector index.
//!
//! MIRAGE ("Mixed Incremental Refinement Approach Graph-based Exploration",
//! Voruganti & Özsu, SIGMOD 2025) is a hierarchical proximity graph
//! index. Its hierarchy is structurally identical to HNSW (level-0 holds
//! every point, levels 1..N hold geometrically decreasing subsets), but
//! Layer 0 is constructed via *refinement-based* iteration rather than
//! incremental greedy insertion:
//!
//! - Layer 0: random `S`-regular graph → repeated rounds of local
//!   RNG-rule pruning + reverse-edge re-injection (paper Algorithm 3).
//! - Layers 1..N: standard HNSW top-down ANNS-driven insertion.
//! - Search: standard HNSW top-down greedy beam search, unmodified.
//!
//! This makes the build path significantly faster than HNSW on most
//! datasets (the paper reports ~6× speedup at N=10M), while preserving
//! HNSW's runtime search performance characteristics.
//!
//! # Implementation strategy in Qdrant
//!
//! Because the resulting graph is *identical in shape* to an HNSW graph,
//! we **compose** [`HNSWIndex`] internally. The MIRAGE build path:
//!
//! 1. Runs the refinement pipeline ([`refinement_builder::build_layer0`])
//!    to produce Layer 0 adjacency.
//! 2. Sets per-point levels using HNSW's standard
//!    `1 / ln M` exponential distribution.
//! 3. Injects Layer 0 into a [`GraphLayersBuilder`] via
//!    [`GraphLayersBuilder::inject_layer0_with_heuristic`].
//! 4. Runs HNSW's [`GraphLayersBuilder::link_new_point_with_min_level`]
//!    with `min_level = 1` to build the upper layers (the search-and-
//!    connect logic is reused unmodified).
//! 5. Persists the graph via [`GraphLayersBuilder::into_graph_layers`]
//!    using the standard HNSW on-disk format (`graph.bin` +
//!    `links*.bin`).
//! 6. Saves a tiny `mirage_config.json` next to it that records the
//!    MIRAGE-specific build parameters (so future opens / config-mismatch
//!    detection know what was built).
//!
//! After build, [`HNSWIndex::open`] is invoked on the same directory to
//! produce a fully-fledged HNSW runtime over the MIRAGE-built graph. All
//! reads (search, telemetry, files, etc.) delegate straight to that
//! inner [`HNSWIndex`].

use std::ops::Deref as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use atomic_refcell::AtomicRefCell;
use common::bitvec::BitSlice;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset, TelemetryDetail};
use fs_err as fs;
use log::debug;
use rand::Rng;
use rayon::ThreadPool;
use rayon::prelude::*;

use super::config::MirageGraphConfig;
use super::refinement_builder::{self, RefinementParams};
use crate::common::operation_error::{OperationResult, check_process_stopped};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorRef};
use crate::id_tracker::{IdTracker, IdTrackerEnum};
use crate::index::VectorIndex;
use crate::index::hnsw_index::HnswM;
use crate::index::hnsw_index::config::HnswGraphConfig;
use crate::index::hnsw_index::graph_layers::GraphLayers;
use crate::index::hnsw_index::graph_layers_builder::GraphLayersBuilder;
use crate::index::hnsw_index::graph_links::GraphLinksFormatParam;
use crate::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
use crate::index::hnsw_index::point_scorer::FilteredScorer;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::segment_constructor::VectorIndexBuildArgs;
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Filter, MirageConfig, SearchParams};
use crate::vector_storage::VectorStorageEnum;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

/// Use the same RNG-based heuristic neighbor selection as HNSW.
///
/// Could be made configurable later, but for parity with HNSW (and the
/// reference MIRAGE implementation, which also uses RNG via FAISS's
/// `shrink_neighbor_list`), we always enable it.
const MIRAGE_USE_HEURISTIC: bool = true;

/// Below this many points, build sequentially to ensure connectivity.
/// Mirrors HNSW's `SINGLE_THREADED_HNSW_BUILD_THRESHOLD`.
#[cfg(debug_assertions)]
const SINGLE_THREADED_MIRAGE_BUILD_THRESHOLD: usize = 32;
#[cfg(not(debug_assertions))]
const SINGLE_THREADED_MIRAGE_BUILD_THRESHOLD: usize = 256;

/// Bytes-in-KB constant used for `full_scan_threshold` conversion.
const BYTES_IN_KB: usize = 1024;

/// MIRAGE-flavored vector index.
///
/// See module docs for the rationale; structurally this just wraps an
/// [`HNSWIndex`].
#[derive(Debug)]
pub struct MirageIndex {
    /// Inner HNSW runtime over the MIRAGE-built graph. Owns id tracker /
    /// vector storage / payload index / quantized vectors / search
    /// telemetry. All [`VectorIndex`] trait methods delegate here.
    inner: HNSWIndex,
    /// Persisted MIRAGE-specific config (`s`, `r`, `iter`,
    /// `num_reverse_edges`, …).
    config: MirageGraphConfig,
    /// Path of the segment directory holding the MIRAGE files.
    path: PathBuf,
}

/// Args for [`MirageIndex::open`] / [`MirageIndex::build`]. Mirrors
/// [`HnswIndexOpenArgs`] so callers in `segment_constructor` can pass the
/// same shared bits straight through.
pub struct MirageIndexOpenArgs<'a> {
    pub path: &'a Path,
    pub id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    pub vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    pub quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    pub payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    pub mirage_config: MirageConfig,
}

impl MirageIndex {
    /// Open an existing MIRAGE index from disk.
    ///
    /// If `mirage_config.json` is missing (e.g. the segment was built by
    /// HNSW and only later configured to use MIRAGE), falls back to
    /// constructing a fresh [`MirageGraphConfig`] from `args.mirage_config`.
    pub fn open(args: MirageIndexOpenArgs<'_>) -> OperationResult<Self> {
        let MirageIndexOpenArgs {
            path,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
            mirage_config,
        } = args;

        let cfg_path = MirageGraphConfig::get_config_path(path);
        let config = if cfg_path.exists() {
            MirageGraphConfig::load(&cfg_path)?
        } else {
            // No persisted config — derive a fresh one from user-facing
            // params + current storage size.
            Self::derive_graph_config(&vector_storage.borrow(), &mirage_config)
        };

        // Hand off to HNSW's open path. Because we save `hnsw_config.json`
        // alongside the graph during build, this picks it up
        // transparently. If only `mirage_config.json` is present (e.g.
        // partial failure), we synthesize a compatible HNSW config.
        let hnsw_config = mirage_config.to_hnsw_compat();
        let inner = HNSWIndex::open(HnswIndexOpenArgs {
            path,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
            hnsw_config,
        })?;

        Ok(MirageIndex {
            inner,
            config,
            path: path.to_owned(),
        })
    }

    /// Build the MIRAGE index from scratch.
    pub fn build<R: Rng + ?Sized>(
        open_args: MirageIndexOpenArgs<'_>,
        build_args: VectorIndexBuildArgs<'_, R>,
    ) -> OperationResult<Self> {
        // Don't allow rebuilding over an already-built index. Mirror HNSW.
        if MirageGraphConfig::get_config_path(open_args.path).exists()
            || HnswGraphConfig::get_config_path(open_args.path).exists()
            || GraphLayers::get_path(open_args.path).exists()
        {
            log::warn!(
                "MIRAGE index already exists at {:?}, skipping building",
                open_args.path
            );
            debug_assert!(false);
            return Self::open(open_args);
        }

        let MirageIndexOpenArgs {
            path,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
            mirage_config,
        } = open_args;
        let VectorIndexBuildArgs {
            permit,
            old_indices: _,
            gpu_device: _,
            rng,
            stopped,
            hnsw_global_config: _,
            feature_flags: _,
            progress: _,
        } = build_args;

        fs::create_dir_all(path)?;

        let id_tracker_ref = id_tracker.borrow();
        let vector_storage_ref = vector_storage.borrow();
        let quantized_vectors_ref = quantized_vectors.borrow();

        let total_vector_count = vector_storage_ref.total_vector_count();
        let deleted_bitslice = vector_storage_ref.deleted_vector_bitslice();

        // Convert KB-based full_scan_threshold to vector-count-based.
        let full_scan_threshold = vector_storage_ref
            .size_of_available_vectors_in_bytes()
            .checked_div(total_vector_count.max(1))
            .and_then(|avg_vector_size| {
                mirage_config
                    .full_scan_threshold
                    .saturating_mul(BYTES_IN_KB)
                    .checked_div(avg_vector_size.max(1))
            })
            .unwrap_or(1);

        let mut config = MirageGraphConfig::new(
            mirage_config.m,
            mirage_config.ef_construct,
            full_scan_threshold,
            mirage_config.s,
            mirage_config.r,
            mirage_config.iter,
            mirage_config.num_reverse_edges,
            mirage_config.max_indexing_threads,
            mirage_config.payload_m,
            total_vector_count,
        );

        let build_main_graph = config.m > 0;
        if !build_main_graph {
            debug!("MIRAGE: m == 0, skipping main graph build");
        }

        // -------- Set up the rayon pool (mirror HNSW for thread priority). --------
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|idx| format!("mirage-build-{idx}"))
            .num_threads(permit.num_cpus as usize)
            .spawn_handler(|thread| {
                let mut b = thread::Builder::new();
                if let Some(name) = thread.name() {
                    b = b.name(name.to_owned());
                }
                if let Some(stack_size) = thread.stack_size() {
                    b = b.stack_size(stack_size);
                }
                b.spawn(|| {
                    #[cfg(target_os = "linux")]
                    if let Err(err) = common::cpu::linux_low_thread_priority() {
                        log::debug!(
                            "Failed to set low thread priority for MIRAGE build, ignoring: {err}"
                        );
                    }
                    thread.run()
                })?;
                Ok(())
            })
            .build()?;

        // -------- Build the graph. --------
        let mut indexed_vectors = 0;
        let graph_layers_builder = if build_main_graph {
            Self::build_main_graph(
                &pool,
                &config,
                stopped,
                rng,
                id_tracker_ref.deref(),
                vector_storage_ref.deref(),
                quantized_vectors_ref.deref(),
                deleted_bitslice,
                &mut indexed_vectors,
            )?
        } else {
            // Empty builder: behaves like a Plain index, but with the
            // MIRAGE config saved so future opens don't try to reinterpret
            // the segment.
            GraphLayersBuilder::new(
                total_vector_count,
                HnswM::new(0, 0),
                config.ef_construct,
                1,
                MIRAGE_USE_HEURISTIC,
            )
        };

        config.indexed_vector_count = Some(indexed_vectors);

        // -------- Persist. --------
        // For Phase 1 we don't support inline-storage; use the standard
        // compressed format on disk and plain in RAM.
        let format_param = GraphLinksFormatParam::Compressed;

        // Always save graph to disk (we'll mmap it back on open). Mirror
        // HNSW's behavior of using `is_on_disk = true` during build to
        // avoid keeping the throw-away builder graph in RAM.
        let graph_layers: GraphLayers =
            graph_layers_builder.into_graph_layers(path, format_param, true)?;
        // We don't need the in-memory graph; HNSWIndex::open will mmap the
        // on-disk version below.
        drop(graph_layers);

        // Save the HNSW-compat config so the inner HNSWIndex can find it.
        config.to_hnsw_compat().save(&HnswGraphConfig::get_config_path(path))?;
        // Save MIRAGE-specific config alongside.
        config.save(&MirageGraphConfig::get_config_path(path))?;

        debug!(
            "MIRAGE build done: {indexed_vectors} indexed, S={s}, R={r}, iter={it}",
            s = config.s,
            r = config.r,
            it = config.iter,
        );

        // Drop the borrow guards before constructing inner HNSWIndex.
        drop(id_tracker_ref);
        drop(vector_storage_ref);
        drop(quantized_vectors_ref);

        // Now hand off to HNSW's open path which will mmap the on-disk
        // graph and wire up scorers / telemetry / quantization.
        let inner = HNSWIndex::open(HnswIndexOpenArgs {
            path,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
            hnsw_config: mirage_config.to_hnsw_compat(),
        })?;

        Ok(MirageIndex {
            inner,
            config,
            path: path.to_owned(),
        })
    }

    /// Heart of the MIRAGE build: produce a fully-populated
    /// [`GraphLayersBuilder`] whose Layer 0 came from refinement and whose
    /// upper layers came from standard HNSW search-and-connect.
    #[allow(clippy::too_many_arguments)]
    fn build_main_graph<R: Rng + ?Sized>(
        pool: &ThreadPool,
        config: &MirageGraphConfig,
        stopped: &AtomicBool,
        rng: &mut R,
        id_tracker_ref: &IdTrackerEnum,
        vector_storage_ref: &VectorStorageEnum,
        quantized_vectors_ref: &Option<QuantizedVectors>,
        deleted_bitslice: &common::bitvec::BitSlice,
        indexed_vectors: &mut usize,
    ) -> OperationResult<GraphLayersBuilder> {
        let total_vector_count = vector_storage_ref.total_vector_count();

        let entry_points_num = std::cmp::max(
            1,
            total_vector_count
                .checked_div(config.full_scan_threshold.max(1))
                .unwrap_or(0)
                * 10,
        );

        let mut builder = GraphLayersBuilder::new(
            total_vector_count,
            HnswM::new(config.m, config.m0),
            config.ef_construct,
            entry_points_num,
            MIRAGE_USE_HEURISTIC,
        );

        // 1) Assign random levels for *all* live points. Same distribution
        //    as HNSW (1/ln M). Done in single-threaded order so subsequent
        //    parallel build sees a stable level table.
        let mut alive_ids: Vec<PointOffsetType> = Vec::with_capacity(total_vector_count);
        for pid in id_tracker_ref
            .point_mappings()
            .iter_internal_excluding(deleted_bitslice)
        {
            *indexed_vectors += 1;
            let level = builder.get_random_layer(rng);
            builder.set_levels(pid, level);
            alive_ids.push(pid);
        }
        debug!("MIRAGE: {} alive points", alive_ids.len());

        // Empty segment: nothing to refine and nothing to link. Return
        // the (empty) builder so the persistence path can still write the
        // standard HNSW files. `HNSWIndex::open` will then load a valid
        // empty graph below.
        if alive_ids.is_empty() {
            debug!("MIRAGE: no live points to index, returning empty builder");
            return Ok(builder);
        }

        // 2) Build Layer 0 adjacency lists via refinement.
        //
        //    `score_pair` is built on top of `RawScorer::score_internal`,
        //    which is symmetric and query-independent. For Phase 1
        //    simplicity we construct a fresh scorer per call — fine for
        //    correctness, but does add a malloc per pairwise distance.
        //    A follow-up should add a per-thread scorer cache (see
        //    `rayon::ThreadLocal` or a `crossbeam::sync::ShardedLock`).
        //
        //    Lifetime/Send story: `vector_storage_ref`, `quantized_vectors_ref`
        //    and `point_deleted` are borrowed for the duration of this
        //    function; they are `Sync`, so `&T` is `Send`, and the closure
        //    we hand to rayon is therefore `Send + Sync`.
        let layer0_adj = {
            // Safe due to the early-return above.
            let entry_pid = alive_ids[0];

            let storage_ref: &VectorStorageEnum = vector_storage_ref;
            let qv_ref: Option<&QuantizedVectors> = quantized_vectors_ref.as_ref();
            let point_deleted: &BitSlice = id_tracker_ref.deleted_point_bitslice();

            let alive_set: std::collections::HashSet<PointOffsetType> =
                alive_ids.iter().copied().collect();

            let score_pair = |a: PointOffsetType, b: PointOffsetType| -> ScoreType {
                let scorer = FilteredScorer::new_internal(
                    entry_pid,
                    storage_ref,
                    qv_ref,
                    None,
                    point_deleted,
                    HardwareCounterCell::disposable(),
                )
                .expect("MIRAGE: failed to build internal scorer");
                scorer.score_internal(a, b)
            };

            let is_alive = |pid: PointOffsetType| -> bool { alive_set.contains(&pid) };

            let params = RefinementParams {
                s: config.s,
                r: config.r,
                iter: config.iter,
                num_reverse_edges: config.num_reverse_edges,
                seed: 2021,
            };

            refinement_builder::build_layer0(
                total_vector_count,
                &params,
                pool,
                is_alive,
                score_pair,
                stopped,
            )?
        };

        // 3) Inject Layer 0 (RNG-prune to m0 slots).
        //
        //    We rebuild a fresh FilteredScorer per call here too, for the
        //    same reason as above. Performance of this phase is dominated
        //    by the heuristic itself, not scorer construction; it can be
        //    optimized later with a per-thread scorer cache.
        check_process_stopped(stopped)?;
        pool.install(|| -> OperationResult<()> {
            (0..total_vector_count)
                .into_par_iter()
                .try_for_each(|u| -> OperationResult<()> {
                    if u % 4096 == 0 {
                        check_process_stopped(stopped)?;
                    }
                    let nbrs = &layer0_adj[u];
                    if nbrs.is_empty() {
                        return Ok(());
                    }
                    let pid = u as PointOffsetType;
                    let scorer = FilteredScorer::new_internal(
                        pid,
                        vector_storage_ref,
                        quantized_vectors_ref.as_ref(),
                        None,
                        id_tracker_ref.deleted_point_bitslice(),
                        HardwareCounterCell::disposable(),
                    )?;
                    builder.inject_layer0_with_heuristic(
                        pid,
                        nbrs.iter().copied(),
                        |a, b| scorer.score_internal(a, b),
                    );
                    Ok(())
                })?;
            Ok(())
        })?;

        // 4) Build upper layers (levels >= 1) with HNSW's standard
        //    search-and-connect.
        check_process_stopped(stopped)?;
        let counter = std::sync::atomic::AtomicU64::new(0);
        let insert_upper = |pid: PointOffsetType| -> OperationResult<()> {
            check_process_stopped(stopped)?;
            let scorer = FilteredScorer::new_internal(
                pid,
                vector_storage_ref,
                quantized_vectors_ref.as_ref(),
                None,
                id_tracker_ref.deleted_point_bitslice(),
                HardwareCounterCell::disposable(),
            )?;
            builder.link_new_point_with_min_level(pid, scorer, 1);
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };

        // Single-threaded warmup, then parallel for the rest, mirroring
        // HNSW's pattern to avoid disconnections in the early entry-point
        // graph.
        let n_warmup = SINGLE_THREADED_MIRAGE_BUILD_THRESHOLD.min(alive_ids.len());
        for &pid in alive_ids[..n_warmup].iter() {
            insert_upper(pid)?;
        }
        if alive_ids.len() > n_warmup {
            pool.install(|| -> OperationResult<()> {
                alive_ids[n_warmup..]
                    .par_iter()
                    .try_for_each(|&pid| insert_upper(pid))
            })?;
        }

        debug!(
            "MIRAGE: built {} upper-layer entries in {:?}",
            counter.load(Ordering::Relaxed),
            std::time::Instant::now()
        );

        Ok(builder)
    }

    /// Construct a [`MirageGraphConfig`] from user-facing config when no
    /// persisted graph config is found on disk. Used by `open` for
    /// segments that were materialized before the MIRAGE config was
    /// introduced (forward-compat).
    fn derive_graph_config(
        vector_storage: &VectorStorageEnum,
        user_cfg: &MirageConfig,
    ) -> MirageGraphConfig {
        let total = vector_storage.available_vector_count().max(1);
        let full_scan_threshold = vector_storage
            .size_of_available_vectors_in_bytes()
            .checked_div(total)
            .and_then(|avg| {
                user_cfg
                    .full_scan_threshold
                    .saturating_mul(BYTES_IN_KB)
                    .checked_div(avg.max(1))
            })
            .unwrap_or(1);
        MirageGraphConfig::new(
            user_cfg.m,
            user_cfg.ef_construct,
            full_scan_threshold,
            user_cfg.s,
            user_cfg.r,
            user_cfg.iter,
            user_cfg.num_reverse_edges,
            user_cfg.max_indexing_threads,
            user_cfg.payload_m,
            total,
        )
    }

    pub fn is_on_disk(&self) -> bool {
        self.inner.is_on_disk()
    }

    pub fn populate(&self) -> OperationResult<()> {
        self.inner.populate()
    }

    pub fn clear_cache(&self) -> OperationResult<()> {
        self.inner.clear_cache()
    }

    pub fn config(&self) -> &MirageGraphConfig {
        &self.config
    }
}

/// User-facing [`crate::types::HnswConfig`] view. Used as the
/// `hnsw_config` field of [`HnswIndexOpenArgs`].
///
/// Note: `on_disk` is propagated from the original [`MirageConfig`] via
/// [`MirageIndex`] (see callers); we don't have the bit here.
impl MirageConfig {
    pub fn to_hnsw_compat(&self) -> crate::types::HnswConfig {
        crate::types::HnswConfig {
            m: self.m,
            ef_construct: self.ef_construct,
            full_scan_threshold: self.full_scan_threshold,
            max_indexing_threads: self.max_indexing_threads,
            on_disk: self.on_disk,
            payload_m: self.payload_m,
            inline_storage: None,
        }
    }
}

impl VectorIndex for MirageIndex {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        // Search delegates verbatim to HNSW. The MIRAGE-built graph is a
        // valid HNSW graph, so this is sound.
        self.inner.search(vectors, filter, top, params, query_context)
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        let mut t = self.inner.get_telemetry_data(detail);
        t.index_name = Some("mirage".to_string());
        t
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut files = self.inner.files();
        let mirage_cfg = MirageGraphConfig::get_config_path(&self.path);
        if mirage_cfg.exists() {
            files.push(mirage_cfg);
        }
        files
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        // Same as HNSW: MIRAGE files are immutable post-build.
        self.files()
    }

    fn indexed_vector_count(&self) -> usize {
        self.config
            .indexed_vector_count
            .unwrap_or_else(|| self.inner.indexed_vector_count())
    }

    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        self.inner.size_of_searchable_vectors_in_bytes()
    }

    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        // For Phase 1, MIRAGE does not support in-place vector updates.
        // The optimizer will rebuild the segment when versions advance.
        // (HNSW also returns an error here.)
        self.inner.update_vector(id, vector, hw_counter)
    }
}
