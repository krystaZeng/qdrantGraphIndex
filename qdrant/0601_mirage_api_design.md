# 0601 Mirage API 设计方案

## 1. 目标

目标是在 Qdrant 中提供用户可调用的 Mirage index 配置入口，使用户能够像选择 HNSW 一样，在 collection/vector 配置中显式选择 Mirage。

P0 阶段不新增独立 search API。搜索仍然复用 Qdrant 现有 search API；是否命中 Mirage 由 collection/vector 的 index 配置、optimizer 构建结果和 segment 内部 `VectorIndexEnum::Mirage` 决定。

## 2. P0 API 形态

Mirage P0 建议暴露在 dense vector 配置中：

```json
{
  "vectors": {
    "size": 128,
    "distance": "Euclid",
    "datatype": "float32",
    "index": {
      "type": "mirage",
      "options": {
        "m": 16,
        "ef_construct": 1024,
        "s": 16,
        "r": 4,
        "iter": 15,
        "num_reverse_edges": 96,
        "max_indexing_threads": 0,
        "on_disk": false
      }
    }
  }
}
```

这里的 `distance` 使用 `Euclid`。在 Qdrant 现有 public API 中，L2/欧氏距离对应的是 `Distance::Euclid`，不建议 P0 新增 `"L2"` 字符串别名。

需要区分两个 `on_disk`：

- `vectors.on_disk`：vector storage 是否使用 mmap/on-disk。
- `index.options.on_disk`：Mirage index 本身是否 on-disk。

P0 阶段这两个字段都不支持 `true`。如果用户显式配置 Mirage 且任一字段为 `true`，应在 validation 阶段拒绝。

## 3. P0 边界

P0 Mirage API 只支持最小闭环：

- dense vector。
- `Float32` vector storage。
- `Euclid` distance。
- no-filter search。
- in-memory Mirage index。
- 无 quantization。
- 无 multivector。
- 无 payload-aware subgraph。

明确不支持：

- `Cosine` / `Dot` / `Manhattan`。
- `float16` / `uint8` vector storage。
- quantization。
- multivector。
- on-disk vector storage。
- on-disk Mirage index。
- `payload_m`。
- filtered search 的 payload-aware subgraph。

如果用户显式配置 Mirage，但请求中出现 P0 不支持的组合，应该在 collection config validation 阶段直接返回 bad input。显式 Mirage 不应该 silent fallback 到 HNSW。

## 4. 为什么不是新增 search API

Mirage 的用户调用方式应该与 HNSW 保持一致：

1. 用户在 collection/vector config 中声明 index 类型。
2. Qdrant optimizer 根据配置构建 optimized segment。
3. segment constructor 构建 `VectorIndexEnum::Mirage`。
4. 普通 search API 走现有 planner 和 vector index 调度。

因此 P0 的 API 工作重点是“配置入口”和“optimizer/segment config 传递链路”，不是新增 `/search_mirage` 之类的旁路接口。

这样可以避免绕开 Qdrant 现有的 collection lifecycle、optimizer、segment persistence、telemetry 和 search path。

## 5. Rust 类型设计

建议在 `lib/collection/src/operations/types.rs` 中新增 public vector index 配置类型：

```rust
#[derive(
    Debug,
    Hash,
    Deserialize,
    Serialize,
    JsonSchema,
    Validate,
    Anonymize,
    Clone,
    PartialEq,
    Eq,
)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type", content = "options")]
pub enum VectorIndexConfig {
    Mirage(MirageConfig),
}
```

然后在 `VectorParams` 中新增字段：

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
#[validate(nested)]
pub index: Option<VectorIndexConfig>,
```

P0 只在 public API 中新增 `Mirage`。HNSW 仍沿用已有 `hnsw_config` 字段，避免一次性重构所有 HNSW API。

这里有一个工程分层取舍：`MirageConfig` 当前定义在 `segment::types`，直接在 collection public API 里复用可以降低 P0 改动面，但会让 public model 直接依赖 segment 内部配置。

P0 可以先复用 `MirageConfig`。后续如果要支持 update collection、config diff 或正式 public API，建议拆出 public 层类型，例如：

```rust
#[derive(
    Debug,
    Hash,
    Deserialize,
    Serialize,
    JsonSchema,
    Validate,
    Anonymize,
    Clone,
    PartialEq,
    Eq,
    Default,
)]
#[serde(rename_all = "snake_case")]
pub struct MirageConfigDiff {
    pub m: Option<usize>,
    pub ef_construct: Option<usize>,
    pub s: Option<usize>,
    pub r: Option<usize>,
    pub iter: Option<usize>,
    pub num_reverse_edges: Option<usize>,
    pub max_indexing_threads: Option<usize>,
    pub on_disk: Option<bool>,
}
```

然后在 collection -> optimizer 阶段把 public `MirageConfigDiff` resolve 成 segment 内部 `MirageConfig`。

## 6. 兼容性规则

已有用户配置保持兼容：

```json
{
  "vectors": {
    "size": 128,
    "distance": "Euclid",
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100
    }
  }
}
```

P0 新规则：

- `index == None`：完全走旧逻辑。
- `index.type == "mirage"`：显式要求“当 segment 达到 indexing threshold 并需要建 dense vector index 时，使用 Mirage 作为目标 backend”。
- `index.type == "mirage"` 不绕过 Qdrant indexing threshold。小 segment 或低于 indexing threshold 的 segment 仍可以保持 `Indexes::Plain {}`。
- `index` 和 legacy `hnsw_config` 同时出现：P0 直接报错，避免语义歧义。
- 显式 Mirage 不依赖 `experimental_mirage_enabled()`。
- experimental flag 可以继续保留作为内部实验入口，但显式 API 优先级更高。

建议错误信息：

```text
`index.type = mirage` cannot be combined with legacy `hnsw_config`.
Use `index.options` for Mirage parameters.
```

## 7. Collection 到 Segment 的配置传递

当前主链路是：

```text
VectorParams
  -> build_segment_optimizer_config
  -> DenseVectorOptimizerInput
  -> DenseVectorOptimizerConfig
  -> resolve_desired_dense_index
  -> VectorDataConfig.index
  -> build_vector_index
  -> VectorIndexEnum::Mirage
```

P0 需要把 `MirageConfig` 从 collection public config 一路传到 shard optimizer。

建议扩展 `lib/shard/src/optimizers/config.rs`：

```rust
pub struct DenseVectorOptimizerConfig {
    pub on_disk: Option<bool>,
    pub hnsw_config: HnswConfig,
    pub mirage_config: Option<MirageConfig>,
    pub quantization_config: Option<QuantizationConfig>,
}
```

同时扩展 `DenseVectorOptimizerInput`：

```rust
pub struct DenseVectorOptimizerInput {
    pub size: usize,
    pub distance: Distance,
    pub on_disk: Option<bool>,
    pub hnsw_config: HnswConfig,
    pub mirage_config: Option<MirageConfig>,
    pub quantization_config: Option<QuantizationConfig>,
    pub multivector_config: Option<MultiVectorConfig>,
    pub datatype: Option<VectorStorageDatatype>,
}
```

在 `build_segment_optimizer_config` 中：

- `VectorParams.index == Some(VectorIndexConfig::Mirage(cfg))` 时，设置 `mirage_config = Some(cfg)`。
- 否则 `mirage_config = None`，保持旧 HNSW 逻辑。

## 8. Index Resolution 修改

当前 `resolve_desired_dense_index` 已支持 experimental Mirage：

```rust
if use_mirage && mirage_p0_unsupported_reason(vector_config, vector_cfg).is_none() {
    Indexes::Mirage(MirageConfig::from_hnsw(vector_cfg.hnsw_config))
} else {
    Indexes::Hnsw(vector_cfg.hnsw_config)
}
```

P0 API 后建议改成：

```rust
if !threshold_is_indexed {
    return Indexes::Plain {};
}

if let Some(mirage_config) = vector_cfg.mirage_config {
    return Indexes::Mirage(mirage_config);
}

if use_mirage && mirage_p0_unsupported_reason(vector_config, vector_cfg).is_none() {
    return Indexes::Mirage(MirageConfig::from_hnsw(vector_cfg.hnsw_config));
}

Indexes::Hnsw(vector_cfg.hnsw_config)
```

这里的 `threshold_is_indexed` 判断必须放在最前面。这意味着即使用户显式配置 Mirage，只要 segment 没有达到 indexing threshold，目标 index 仍然是 `Indexes::Plain {}`。这个语义是合理的：`index.type = mirage` 指定的是“需要建索引时使用 Mirage backend”，不是强制小数据量立即建 Mirage。

需要明确区分三种来源：

1. explicit Mirage config
   - 合法则 target = `Indexes::Mirage(cfg)`。
   - 非法应早在 collection validation 阶段拒绝。
   - 不 silent fallback 到 HNSW。

2. experimental Mirage flag
   - 满足 P0 guard 才 target = Mirage。
   - 不满足则 fallback HNSW。

3. default
   - target = HNSW。

`ConfigMismatchOptimizer` 必须使用同一个 resolver，不能自己复制一套 HNSW/Mirage 判断逻辑。否则 create/build path 和 mismatch/optimize path 可能产生不同 target，导致同一个 collection 内 HNSW 和 Mirage 混跑，污染 benchmark 和行为判断。

## 9. Validation 设计

建议新增 collection-level validation，例如：

```rust
fn validate_vector_index_config(params: &VectorParams) -> CollectionResult<()> {
    let Some(VectorIndexConfig::Mirage(mirage)) = params.index else {
        return Ok(());
    };

    if params.hnsw_config.is_some() {
        return Err(CollectionError::bad_input(
            "`index.type = mirage` cannot be combined with legacy `hnsw_config`. Use `index.options` for Mirage parameters.",
        ));
    }

    if params.distance != Distance::Euclid {
        return Err(CollectionError::bad_input(
            "Mirage P0 supports Euclid distance only",
        ));
    }

    if params.datatype.is_some_and(|datatype| datatype != Datatype::Float32) {
        return Err(CollectionError::bad_input(
            "Mirage P0 supports Float32 vector storage only",
        ));
    }

    if params.quantization_config.is_some() {
        return Err(CollectionError::bad_input(
            "Mirage P0 does not support quantization",
        ));
    }

    if params.multivector_config.is_some() {
        return Err(CollectionError::bad_input(
            "Mirage P0 does not support multivector",
        ));
    }

    if params.on_disk == Some(true) || mirage.on_disk == Some(true) {
        return Err(CollectionError::bad_input(
            "Mirage P0 does not support on-disk vector storage or on-disk Mirage index",
        ));
    }

    if mirage.payload_m.is_some() {
        return Err(CollectionError::bad_input(
            "Mirage P0 does not support payload-aware subgraphs",
        ));
    }

    Ok(())
}
```

Collection validation 只能处理配置级限制，例如 distance、datatype、quantization、multivector、on-disk、payload_m。

请求级限制必须保留在 runtime search path。例如 no-filter 限制不能只靠 collection validation，因为 search request 才知道是否带 filter。`MirageIndex::search` 中必须保留类似逻辑：

```rust
if filter.is_some() {
    return Err(OperationError::ValidationError {
        description: "Mirage P0 supports no-filter search only".to_string(),
    });
}
```

Segment 层已有 Mirage P0 config validation 也应保留，作为内部防线。

## 10. REST / OpenAPI

因为 `VectorParams` 使用 `Serialize` / `Deserialize` / `JsonSchema`，新增 `index` 字段后，REST schema 应该可以通过现有 schema 生成链路暴露。

需要更新或验证：

- REST model schema。
- OpenAPI snapshot / consistency check。
- 文档示例使用 `"distance": "Euclid"`。
- 错误信息能清楚说明 P0 限制。

P0 不建议新增 `"L2"` alias。若未来要兼容 `"L2"`，应单独设计为 `Distance` 反序列化别名，不和 Mirage API 首版混在一起。

## 11. gRPC 策略

如果 P0 定位为 internal / experimental，REST/API model 先闭环、不立刻同步 gRPC 可以接受。

如果要作为正式 Qdrant public API 发布，REST 和 gRPC 必须同步。长期 REST/gRPC 不一致会造成 SDK、客户端和兼容性风险。

如果必须同步 gRPC，需要修改：

- `lib/api/src/grpc/proto/collections.proto`
- 生成后的 `lib/api/src/grpc/qdrant.rs`
- `lib/collection/src/operations/conversions.rs`

这会显著扩大改动面。建议先完成 REST/create collection 路径和 segment 行为验证，再补 gRPC。正式发布前必须补齐 gRPC。

## 12. 测试计划

最低测试集：

1. 反序列化测试
   - `index.type = "mirage"` 能解析成 `VectorIndexConfig::Mirage`。
   - `"distance": "Euclid"` 正常解析。

2. validation 正例
   - dense + Float32 + Euclid + no quantization + no multivector + no on-disk。

3. validation 负例
   - `Cosine` / `Dot` / `Manhattan`。
   - `float16` / `uint8`。
   - quantization。
   - multivector。
   - `on_disk = true`。
   - `mirage.options.on_disk = true`。
   - `payload_m`。
   - `index` 和 `hnsw_config` 同时出现。

4. optimizer config 传递测试
   - `VectorParams.index = Mirage(cfg)` 能传到 `DenseVectorOptimizerConfig.mirage_config`。

5. index resolution 测试
   - 显式 Mirage 生成 `Indexes::Mirage(cfg)`。
   - 不依赖 `experimental_mirage_enabled()`。
   - threshold 未达到时仍是 `Indexes::Plain {}`。
   - explicit Mirage + vector size below indexing threshold => target `Indexes::Plain {}`，并且不应误判为 API 未生效。

6. experimental flag 与 explicit API 优先级测试
   - env off + explicit Mirage => target Mirage。
   - env on + no explicit config + eligible => target Mirage。
   - env on + explicit Mirage invalid config => validation reject，不 fallback。

7. runtime filtered search reject 测试
   - create collection with Mirage。
   - insert vectors。
   - search with filter。
   - 返回 validation error：`Mirage P0 supports no-filter search only`。

8. collection-level 集成测试
   - create collection with Mirage + Euclid + Float32。
   - insert 2000 vectors。
   - 触发 optimize/build index。
   - no-filter search。
   - 验证最终 segment index 是 `VectorIndexEnum::Mirage`。
   - recall@100 合理，例如继续参考当前 segment 测试中的 `0.95+` 级别。

## 13. 验收标准

P0 完成后，用户应该能通过普通 Qdrant create collection API 指定 Mirage：

```json
{
  "vectors": {
    "size": 128,
    "distance": "Euclid",
    "datatype": "float32",
    "index": {
      "type": "mirage",
      "options": {
        "m": 16,
        "ef_construct": 1024,
        "s": 16,
        "r": 4,
        "iter": 15,
        "num_reverse_edges": 96,
        "max_indexing_threads": 0,
        "on_disk": false
      }
    }
  }
}
```

随后用户继续调用普通 search API。只要 collection/segment 已优化到 Mirage index，搜索链路应该命中 `VectorIndexEnum::Mirage`，而不是要求用户直接调用 `MirageIndex::build`。

注意：如果 segment 未达到 indexing threshold，最终 index 仍然可以是 `Indexes::Plain {}`。这不是 Mirage API 失效，而是 Qdrant 的正常 threshold 语义。

## 14. 方案边界判断

如果目标是“内部 experimental public config P0”，当前方案可以推进：

- REST/API model 先闭环。
- explicit Mirage 不依赖 experimental flag。
- illegal config 直接 validation reject。
- no-filter 在 runtime search guard 拒绝。
- indexing threshold 语义保留。

如果目标是“正式对外 public API”，还需要补齐：

- gRPC proto / conversion。
- update collection。
- config diff。
- OpenAPI snapshot。
- 文档。
- SDK/客户端兼容策略。
- public `MirageConfigDiff` 或 `MirageIndexParams` 与 segment internal `MirageConfig` 的分层。

核心原则：

1. Mirage 是 index backend，不是新 search API。
2. 显式 Mirage 不 silent fallback。
3. 显式 Mirage 不绕过 indexing threshold。
4. P0 能力边界要在 validation 和 search guard 两层封住。
5. experimental flag 和 explicit API 要区分。
6. collection config -> optimizer -> segment 的链路要完整打通。

## 15. 当前实现基础

当前已经具备开始做 API 的基础：

- `Indexes::Mirage(MirageConfig)` 已存在。
- `MirageConfig` 已有默认值和 C++ 对齐参数。
- segment constructor 已能从 `Indexes::Mirage` 分发到 `VectorIndexEnum::Mirage`。
- `MirageIndex::build/open/search` 已通过 segment-level 测试。
- authoritative C++ golden 已覆盖 Layer0 refinement / Layer0 injection / persisted GraphLayers。
- no-filter segment integration 当前 recall@100 为 `0.959`。

因此下一步可以开始做 public API 配置入口，重点是 collection config、optimizer config、validation 和 REST/OpenAPI 测试。
