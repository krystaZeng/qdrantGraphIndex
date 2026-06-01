# Mirage 测试方法设计

日期：2026-06-01

## 背景

根据 `0601_architecture.md` 的结论，Mirage 和 HNSW 在 Qdrant 的索引抽象层是并列关系：

```rust
Indexes::Hnsw(...)
Indexes::Mirage(...)

VectorIndexEnum::Hnsw(...)
VectorIndexEnum::Mirage(...)
```

但在 MirageIndex 的内部实现层，Mirage 组合并复用了 HNSWIndex：

```text
VectorIndexEnum
├── Hnsw(HNSWIndex)
└── Mirage(MirageIndex)
        └── inner: HNSWIndex
```

因此测试也需要分成两类：

1. MirageIndex 本体测试：直接调用 `MirageIndex::build(...)`，与 `HNSWIndex::build(...)` 平行。
2. Qdrant enum / constructor 分发测试：通过 `Indexes::Mirage(...)` 触发 segment constructor，断言最终得到 `VectorIndexEnum::Mirage(...)`。

这两类测试验证的是不同层级，不应该混成一个断言目标。

## 测试文件

建议新增：

```text
lib/segment/tests/integration/mirage_index_test.rs
```

并在：

```text
lib/segment/tests/integration/main.rs
```

注册：

```rust
mod mirage_index_test;
```

## 公共参数

建议先使用中等规模数据，避免测试过慢：

```rust
const DIM: usize = 128;
const NUM_VECTORS: u64 = 2_000;
const TOP: usize = 100;
const ATTEMPTS: usize = 50;
const SEARCH_EF: usize = 256;
const MIN_AVG_RECALL: f64 = 0.70;
```

`MirageConfig` 建议固定为：

```rust
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
```

说明：

- `ef_construct = 1024` 与 C++ Mirage upper layer build 行为对齐。
- `full_scan_threshold = 1` 是为了尽量避免小数据集直接走 full scan，从而让测试真正覆盖 graph search。
- `max_indexing_threads = 1` 用于降低测试中的非确定性。
- `on_disk = Some(false)`、`payload_m = None` 保持 Mirage P0 范围。

## 测试一：MirageIndex 本体测试

### 目标

验证 `MirageIndex` 作为 index-level 实现，能像 `HNSWIndex` 一样直接 build、open、search。

这个测试不验证 `Indexes::Mirage -> VectorIndexEnum::Mirage` 的 enum 分发路径。

### 测试名

```rust
#[test]
fn mirage_index_direct_build_open_search_recall()
```

### 核心路径

```text
build_simple_segment(Plain)
-> upsert 2000 dense Float32 vectors
-> MirageIndex::build(...)
-> assert indexed_vector_count
-> assert mirage_config.json exists
-> MirageIndex::open(...)
-> no-filter search
-> plain vector_index search
-> recall@10
```

### 关键实现

构建 plain source segment：

```rust
let stopped = AtomicBool::new(false);
let hw_counter = HardwareCounterCell::new();
let mut data_rng = StdRng::seed_from_u64(42);

let segment_dir = Builder::new().prefix("mirage_source").tempdir().unwrap();
let mirage_dir = Builder::new().prefix("mirage_index").tempdir().unwrap();

let mut segment = build_simple_segment(segment_dir.path(), DIM, Distance::Euclid).unwrap();

for n in 0..NUM_VECTORS {
    let vector = random_vector(&mut data_rng, DIM);
    segment
        .upsert_point(
            n as SeqNumberType,
            n.into(),
            only_default_vector(&vector),
            &hw_counter,
        )
        .unwrap();
}
```

直接 build MirageIndex：

```rust
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
```

基础断言：

```rust
assert_eq!(mirage_index.indexed_vector_count(), NUM_VECTORS as usize);
assert!(mirage_dir.path().join("mirage_config.json").exists());
```

重新 open：

```rust
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
```

搜索并对比 plain/exact：

```rust
let query = random_vector(&mut query_rng, DIM).into();

let exact = segment.vector_data[DEFAULT_VECTOR_NAME]
    .vector_index
    .borrow()
    .search(&[&query], None, TOP, None, &Default::default())
    .unwrap()
    .remove(0);

let approx = mirage_index
    .search(
        &[&query],
        None,
        TOP,
        Some(&SearchParams {
            hnsw_ef: Some(SEARCH_EF),
            ..Default::default()
        }),
        &Default::default(),
    )
    .unwrap()
    .remove(0);
```

建议将结果转换成 external id 后算 recall，避免不同 segment 场景下 internal id 不一致：

```rust
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
```

累计 recall：

```rust
let exact_ids = result_external_ids(&segment, &exact);
let approx_ids = result_external_ids(&segment, &approx);
let overlap = exact_ids.intersection(&approx_ids).count();
let recall = overlap as f64 / TOP as f64;
```

最终断言：

```rust
assert!(
    avg_recall >= MIN_AVG_RECALL,
    "Mirage recall@10 too low: {avg_recall:.3}, expected >= {MIN_AVG_RECALL:.3}"
);
```

## 测试二：Qdrant enum / constructor 分发测试

### 目标

验证 Qdrant segment 配置路径能把：

```rust
Indexes::Mirage(...)
```

通过 segment constructor 构建成：

```rust
VectorIndexEnum::Mirage(...)
```

这个测试验证的是 Qdrant enum / constructor 分发路径，不是 Mirage 构建算法本体。

### 测试名

```rust
#[test]
fn mirage_segment_constructor_builds_vector_index_enum_mirage()
```

### 为什么通过 SegmentBuilder

`build_vector_index(...)` 是 `pub(crate)`，integration test 不能直接调用。

因此需要通过：

```rust
SegmentBuilder::new(...)
builder.update(...)
builder.build(...)
```

间接触发 `build_vector_index(...)`。

### 核心路径

```text
source segment: Plain
-> upsert 1000-2000 vectors
-> target SegmentConfig { index: Indexes::Mirage(...) }
-> SegmentBuilder::new(...)
-> builder.update(&[&source])
-> builder.build(...)
-> built_segment.vector_data[DEFAULT_VECTOR_NAME].vector_index
-> assert VectorIndexEnum::Mirage
-> optional no-filter search sanity
```

### target SegmentConfig

```rust
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
```

### 关键实现

构建 source segment：

```rust
let source_dir = Builder::new().prefix("mirage_constructor_source").tempdir().unwrap();
let output_dir = Builder::new().prefix("mirage_constructor_output").tempdir().unwrap();
let temp_dir = Builder::new().prefix("mirage_constructor_temp").tempdir().unwrap();

let mut source_segment = build_simple_segment(source_dir.path(), DIM, Distance::Euclid).unwrap();

for n in 0..NUM_VECTORS {
    let vector = random_vector(&mut data_rng, DIM);
    source_segment
        .upsert_point(
            n as SeqNumberType,
            n.into(),
            only_default_vector(&vector),
            &hw_counter,
        )
        .unwrap();
}
```

通过 SegmentBuilder 触发 constructor：

```rust
let mut builder =
    SegmentBuilder::new(
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
```

断言 enum 分发结果：

```rust
let vector_index = built_segment.vector_data[DEFAULT_VECTOR_NAME]
    .vector_index
    .borrow();

match &*vector_index {
    VectorIndexEnum::Mirage(index) => {
        assert_eq!(index.indexed_vector_count(), NUM_VECTORS as usize);
    }
    other => panic!("expected VectorIndexEnum::Mirage, got {other:?}"),
}
```

继续执行 no-filter search，并与 source plain segment 计算 recall@10：

```rust
let avg_recall =
    average_recall_against_plain(&source_segment, &built_segment, 100, |query| {
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

assert!(
    avg_recall >= MIN_AVG_RECALL,
    "Mirage constructor recall@10 too low: {avg_recall:.3}, expected >= {MIN_AVG_RECALL:.3}"
);
```

## 推荐导入

测试文件大致需要：

```rust
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use common::types::{ExtendedPointId, ScoredPointOffset};
use rand::prelude::StdRng;
use rand::SeedableRng;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::entry_point::SegmentEntry;
use segment::fixtures::payload_fixtures::random_vector;
use segment::index::mirage_index::{MirageIndex, MirageIndexOpenArgs};
use segment::index::{VectorIndex, VectorIndexEnum};
use segment::segment::Segment;
use segment::segment_constructor::VectorIndexBuildArgs;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Distance, HnswGlobalConfig, Indexes, MirageConfig, SearchParams, SegmentConfig,
    SeqNumberType, VectorDataConfig, VectorStorageType,
};
use tempfile::Builder;
use uuid::Uuid;
```

如果使用 `builder.build(...)`，还需要确认 `ProgressTracker::new_for_test()` 和相关 test helper 在 integration test feature 下可用；现有 integration tests 已经大量使用该写法。

## 运行命令

只跑 Mirage integration 测试：

```bash
cargo test -p segment --test integration mirage_index -- --nocapture
```

只跑 direct build 测试：

```bash
cargo test -p segment --test integration mirage_index_direct_build_open_search_recall -- --nocapture
```

只跑 constructor 分发测试：

```bash
cargo test -p segment --test integration mirage_segment_constructor_builds_vector_index_enum_mirage -- --nocapture
```

## 验收标准

第一类测试通过后，可以说明：

- `MirageIndex::build(...)` 能直接构建 Mirage index。
- `MirageIndex::open(...)` 能重新加载已持久化的 Mirage index。
- `MirageIndex::search(...)` 能执行 no-filter dense Float32 search。
- Mirage 搜索结果与 plain/exact search 有可解释的 recall@10。

第二类测试通过后，可以说明：

- `Indexes::Mirage(...)` 能通过 segment constructor 进入 Mirage 构建分支。
- 构建结果确实是 `VectorIndexEnum::Mirage(...)`。
- Qdrant segment config / constructor 层已经承认 Mirage 是与 HNSW 并列的 index 类型。

这两类测试都通过后，才能比较完整地说明 Mirage P0 已覆盖“index 本体可用”和“Qdrant 分发路径可用”两个层级。
