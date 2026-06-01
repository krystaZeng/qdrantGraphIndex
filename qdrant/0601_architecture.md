# Mirage 与 HNSW 的架构关系

日期：2026-06-01

## 结论

当前实现里，Mirage Index 和 HNSW Index 需要分层理解。

在 Qdrant 的索引抽象层，Mirage 和 HNSW 是并列关系：

```rust
Indexes::Hnsw(...)
Indexes::Mirage(...)

VectorIndexEnum::Hnsw(...)
VectorIndexEnum::Mirage(...)
```

也就是说，从 segment config、segment constructor、runtime enum 的角度看，Mirage 是和 HNSW 平级的一个 index 类型。

但在 MirageIndex 的内部实现层，Mirage 不是完全独立实现一套 runtime/search/persistence，而是组合并复用了 HNSWIndex。

当前 Rust 实现的结构可以理解为：

```rust
pub struct MirageIndex {
    inner: HNSWIndex,
    config: MirageGraphConfig,
    path: PathBuf,
}
```

也就是：

```text
VectorIndexEnum
├── Hnsw(HNSWIndex)
└── Mirage(MirageIndex)
        └── inner: HNSWIndex
```

这并不表示 Mirage 不是一个独立索引类型，而是表示 Mirage 构建出来的图在结构上是 HNSW-shaped graph，因此 runtime search、graph persistence、telemetry 等能力可以复用已有 HNSWIndex。

## 与 C++ 参考实现的关系

C++ 参考实现也是类似思路。

`IndexMirage` 内部同时持有：

```cpp
faiss::IndexHNSWFlat hierarchy;
Mirage mirage;
```

核心流程是：

1. `mirage.build(...)` 构建 MIRAGE 的 Layer 0。
2. `add_levels(...)` 构建 HNSW upper layers。
3. `init_level_0_from_knngraph(...)` 把 Mirage 的 Layer 0 注入 HNSW hierarchy。
4. `search(...)` 调用 `hierarchy.search(...)`。

因此更准确的表述是：

```text
Mirage 和 HNSW 在 Qdrant 索引类型上并列；
Mirage 在内部复用 HNSW 作为 graph runtime。
```

## 测试分层

基于这个架构，测试也应该分成两类。

第一类是 MirageIndex 本体测试，直接调用：

```rust
MirageIndex::build(...)
```

这类测试和现有直接调用 `HNSWIndex::build(...)` 的测试平行，用来验证 Mirage 本体能 build、open、search，并能和 plain/exact search 对比 recall。

第二类是 Qdrant enum / constructor 分发测试，通过：

```rust
Indexes::Mirage(...)
```

触发 `build_vector_index(...)`，并断言最终得到：

```rust
VectorIndexEnum::Mirage(...)
```

这类测试验证的是 Qdrant segment 配置分发路径，不是 Mirage 构建算法本体。

## 当前判断

如果目标是“与 HNSW index-level 测试平行”，应该优先直接调用 `MirageIndex::build(...)`。

如果目标是“验证 Qdrant 已经能通过 segment config 使用 Mirage”，则应该通过 `Indexes::Mirage(...)` 走 segment constructor 或 SegmentBuilder。

两类测试都需要，但它们验证的是不同层级，不应该混在一个断言目标里。
