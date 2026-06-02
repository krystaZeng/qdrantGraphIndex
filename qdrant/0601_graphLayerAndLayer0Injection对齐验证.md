# GraphLayers / Layer 0 Injection 对齐验证方案

日期：2026-06-01

## 1. 核心判断

这一步不要一上来比较完整 HNSW search 结果。应该先验证最关键、最局部的语义：

```text
C++ init_level_0_from_knngraph(k = S)
是否等价于
Rust GraphLayersBuilder::inject_layer0_with_heuristic(...take(S))
```

这本质上是在比较 Layer 0 candidate list 经过 C++ HNSW 真实 `init_level_0_from_knngraph(...)` 注入之后，最终写入 HNSW Layer 0 slot 的邻接表，是否与 Rust `inject_layer0_with_heuristic(...)` 的输出一致。

这里有一个关键原则：

```text
layer0_injected 的 authoritative golden 必须来自 C++ 真实执行结果，
而不是 dumper 里重新手写一份 shrink_neighbor_list 模拟逻辑。
```

原因是我们要验证 Rust 是否对齐 C++ 真实实现，而不是验证 Rust 是否对齐“另一份手写 C++ 模拟实现”。

## 2. 要验证的精确对象

C++ 路径是：

```cpp
hierarchy.init_level_0_from_knngraph(k, D.data(), I.data());
```

内部逻辑是：

```cpp
for each point i:
    initial_list = top-k candidates from MIRAGE final_graph
    shrink_neighbor_list(qdis, initial_list, shrunk_list, dest_size)
    write shrunk_list into hnsw.neighbors layer 0
```

Rust 路径是：

```rust
builder.inject_layer0_with_heuristic(
    pid,
    nbrs.iter().copied().take(config.s),
    |a, b| scorer.score_internal(a, b),
);
```

内部逻辑是：

```rust
LinksContainer::fill_from_sorted_with_heuristic(candidates, m0, score)
```

所以最小可验证等价关系是：

```text
C++ hierarchy.init_level_0_from_knngraph(k = S, D, I)
执行后 hnsw.neighbors 中的 Layer 0 links
==
Rust inject_layer0_with_heuristic(top S candidates, level_m = m0)
```

注意：这里比较的是 directed outbound links，不是双向边。`init_level_0_from_knngraph` 本身也是按每个点写自己的 Layer 0 neighbor slot，不会额外补 reverse edge。

`HNSW::shrink_neighbor_list(...)` 是 C++ `init_level_0_from_knngraph(...)` 内部会调用的关键函数，但 golden fixture 不应该通过在 dumper 中复刻它来生成。复刻版 shrink 可以保留为辅助诊断工具，用于 mismatch 时定位 priority queue、tie、distance 方向等问题，但不能作为唯一 golden 来源。

## 3. 改动一：扩展 C++ golden fixture

文件：

```text
tools/mirage_golden/dump_mirage_golden.cpp
```

这里需要调整工具定位：如果要生成 authoritative `layer0_injected`，dumper 应该调用真实 C++ FAISS/Mirage 实现，而不是完全自包含地复刻所有逻辑。

推荐两种实现方式：

1. 在本地 C++ reference 工程里新增一个 fixture dump 工具，直接使用真实 `faiss::Mirage`、`faiss::IndexHNSWFlat` / `faiss::IndexHNSW`、`IndexHNSW::init_level_0_from_knngraph(...)`。
2. 或者让 `tools/mirage_golden/dump_mirage_golden.cpp` 链接本地 FAISS/Mirage 源码，真实调用 `init_level_0_from_knngraph(...)`。

不推荐把 `layer0_injected` 继续建立在 dumper 里手写的 `shrink_layer0_for_point(...)` 上。

当前它已经输出：

```json
{
  "layer0": [...],
  "levels_zero_based": [...],
  "upper_insert_order_by_level": {...}
}
```

下一步增加字段：

```json
{
  "params": {
    "s": 16,
    "hnsw_m": 16,
    "hnsw_m0": 32,
    "layer0_k": 16
  },
  "layer0_top_s": [...],
  "layer0_injected": [...],
  "layer0_injected_stats": {
    "total_edges": 0,
    "min_degree": 0,
    "max_degree": 0,
    "avg_degree": 0.0,
    "degree_histogram": {},
    "reciprocal_edges": 0,
    "reciprocity_ratio": 0.0,
    "weak_connected_components": 0,
    "largest_weak_component": 0
  }
}
```

字段语义：

- `layer0`：C++ Mirage refinement 最终输出，已经有。
- `layer0_top_s`：每个点取 `layer0[u][0..S]`。
- `layer0_injected`：真实调用 C++ `hierarchy.init_level_0_from_knngraph(k = S, D, I)` 后，从 `hierarchy.hnsw.neighbors` 的 Layer 0 slot 读取出来的最终 neighbor list。
- `layer0_injected_stats`：用于诊断整体图形态。

authoritative fixture 生成流程应该是：

```text
1. 运行真实 C++ Mirage::build(...)
   得到 mirage.offsets / mirage.final_graph

2. 构造 D / I
   k = mirage.S
   I[u * k + j] = mirage.final_graph[mirage.offsets[u] + j]
   D[u * k + j] = distance(u, I[u * k + j])

3. 调用真实 C++ HNSW injection
   hierarchy.init_level_0_from_knngraph(k, D.data(), I.data())

4. 从真实 C++ HNSW graph 中读取 Layer 0
   hierarchy.hnsw.neighbor_range(u, 0, &begin, &end)
   hierarchy.hnsw.neighbors[begin..end]
   过滤 -1

5. 将读取结果写入 fixture.layer0_injected
```

这里 `max_size` 应该和 Rust `m0` 对齐。当前 Mirage config 是 `m = 16`，所以 `m0 = 32`。

但由于 `k = S = 16`，实际输出最多 16 条边。也就是说：

```text
m0 = 32
k = 16
最终 degree <= 16
```

这一点和 C++ `IndexMirage::convert_to_mirage` 一致，因为它传入的是：

```cpp
int k = mirage.S;
hierarchy.init_level_0_from_knngraph(k, D, I);
```

可以额外保留一个 dumper 内部的 `shrink_layer0_for_point(...)` 作为 diagnostic output，例如：

```json
{
  "layer0_injected_diagnostic_shrink": [...]
}
```

但它只能用于 mismatch 排查，不能替代真实 `hierarchy.hnsw.neighbors` dump 出来的 `layer0_injected`。

## 4. 改动二：Rust 侧新增 exact parity test

建议先做两个层次的测试。

### 4.1 低层 heuristic parity test

目标：验证 Rust `LinksContainer::fill_from_sorted_with_heuristic` 对同一组 top-S candidates 的输出，是否与真实 C++ `init_level_0_from_knngraph(...)` 注入后的 `layer0_injected` 一致。

这里的 `fixture.layer0_injected[u]` 必须来自 C++ 真实 `hierarchy.hnsw.neighbors`，不能来自 dumper 中手写 shrink 模拟。

位置建议：

```text
lib/segment/src/index/hnsw_index/links_container.rs
```

或者放在 Mirage golden test 中，但最好靠近 `LinksContainer`，因为被测对象就是 heuristic shrink。

测试逻辑：

```rust
#[test]
fn test_layer0_injection_heuristic_matches_cpp_golden() {
    let fixture = load_cpp_golden_n64_d8_l2();
    let vectors = generate_vectors(&fixture);

    for u in 0..fixture.n {
        let candidates = fixture.layer0[u]
            .iter()
            .copied()
            .take(fixture.params.s)
            .map(|idx| ScoredPointOffset {
                idx: idx as PointOffsetType,
                score: -l2(&vectors[u], &vectors[idx]),
            });

        let mut links = LinksContainer::with_capacity(fixture.params.hnsw_m0);

        links.fill_from_sorted_with_heuristic(
            candidates,
            fixture.params.hnsw_m0,
            |a, b| -l2(&vectors[a as usize], &vectors[b as usize]),
        );

        let actual: Vec<usize> = links
            .links()
            .iter()
            .map(|&idx| idx as usize)
            .collect();

        assert_eq!(actual, fixture.layer0_injected[u]);
    }
}
```

这个测试非常关键，因为它直接证明：

```text
真实 FAISS IndexHNSW::init_level_0_from_knngraph(...)
在 Layer 0 写出的 links
==
Qdrant LinksContainer::fill_from_sorted_with_heuristic
```

在 score/distance 方向转换后语义一致。

### 4.2 GraphLayersBuilder injection parity test

目标：验证 Mirage 使用的实际 builder 入口没有引入偏差。

位置建议：

```text
lib/segment/src/index/hnsw_index/graph_layers_builder.rs
```

需要加一个 test-only accessor：

```rust
#[cfg(test)]
pub(crate) fn raw_links_for_test(
    &self,
    point_id: PointOffsetType,
    level: usize,
) -> Vec<PointOffsetType> {
    self.links_layers[point_id as usize][level]
        .read()
        .links()
        .to_vec()
}
```

测试逻辑：

```rust
#[test]
fn test_graph_layers_builder_layer0_injection_matches_cpp_golden() {
    let fixture = load_cpp_golden_n64_d8_l2();
    let vectors = generate_vectors(&fixture);

    let builder = GraphLayersBuilder::new(
        fixture.n,
        HnswM::new(fixture.params.hnsw_m, fixture.params.hnsw_m0),
        1024,
        1,
        true,
    );

    for u in 0..fixture.n {
        let candidates = fixture.layer0[u]
            .iter()
            .copied()
            .take(fixture.params.s)
            .map(|idx| ScoredPointOffset {
                idx: idx as PointOffsetType,
                score: -l2(&vectors[u], &vectors[idx]),
            });

        builder.inject_layer0_with_heuristic(
            u as PointOffsetType,
            candidates,
            |a, b| -l2(&vectors[a as usize], &vectors[b as usize]),
        );

        let actual: Vec<usize> = builder
            .raw_links_for_test(u as PointOffsetType, 0)
            .into_iter()
            .map(|idx| idx as usize)
            .collect();

        assert_eq!(actual, fixture.layer0_injected[u]);
    }
}
```

这个测试覆盖的是：

```text
Mirage 使用的实际注入 API
```

而不是只测底层 container。

## 5. 改动三：端到端 build 后读取 GraphLayers

前两个是局部 exact test。之后再补一个真实 build-path 测试，验证 `MirageIndex::build` 最终持久化出来的 Layer 0 没有变形。

位置建议：

```text
lib/segment/tests/integration/mirage_graph_layers_test.rs
```

流程：

1. 用 fixture 里的 deterministic vectors 构建 segment。
2. 直接调用 `MirageIndex::build(...)`。
3. 从 Mirage index 目录读取 `GraphLayers`：

```rust
let graph = GraphLayers::load(mirage_dir.path(), false, false)?;
```

4. 对每个点读取 level 0 links：

```rust
let mut links = Vec::new();
graph.for_each_link(pid, 0, |link| links.push(link));
```

5. 和 `fixture.layer0_injected[pid]` 比较。

如果完全 exact 能过，最好：

```rust
assert_eq!(actual, fixture.layer0_injected);
```

如果因为最终 materialization 有合法重排，则改成集合比较：

```rust
assert_eq!(HashSet(actual), HashSet(expected));
```

但第一版应该先追求 exact。当前 fixture 是小数据、单线程、无删除点，理论上可以 exact。

## 6. 统计指标实现

新增一个 test helper：

```rust
struct Layer0Stats {
    total_edges: usize,
    min_degree: usize,
    max_degree: usize,
    avg_degree: f64,
    degree_histogram: BTreeMap<usize, usize>,
    reciprocal_edges: usize,
    reciprocity_ratio: f64,
    weak_connected_components: usize,
    largest_weak_component: usize,
}
```

输入：

```rust
Vec<Vec<usize>>
```

计算规则：

- `degree[u] = graph[u].len()`
- `total_edges = sum(degree)`
- `degree_histogram[degree] += 1`
- `reciprocal_edges`：对每条 directed edge `u -> v`，如果存在 `v -> u`，计数。
- `reciprocity_ratio = reciprocal_edges / total_edges`
- `weak_connected_components`：把 directed edge 当 undirected edge 做 BFS/Union-Find。
- `largest_weak_component`：最大弱连通分量大小。

验收建议：

第一阶段 exact：

```rust
assert_eq!(rust_stats, fixture.layer0_injected_stats);
```

如果 exact graph 已经比较通过，stats 其实是冗余的，但它在失败时非常有诊断价值。

## 7. 推荐落地顺序

建议按这个顺序做：

1. 改造 C++ golden dumper，使 `layer0_injected` 来自真实 C++ HNSW injection
   - 运行真实 `faiss::Mirage::build(...)`
   - 构造 `D / I`
   - 调用真实 `hierarchy.init_level_0_from_knngraph(k = S, D, I)`
   - 从 `hierarchy.hnsw.neighbors` 读取 Layer 0 slot
   - 过滤 `-1`
   - 写入 `layer0_injected`

2. 扩展 fixture 输出字段
   - 输出 `hnsw_m0`
   - 输出 `layer0_top_s`
   - 输出 `layer0_injected`
   - 输出 `layer0_injected_stats`
   - 可选输出 `layer0_injected_diagnostic_shrink`，只用于 mismatch 调试

3. 重新生成 fixture：

   ```bash
   clang++ -std=c++17 -O2 tools/mirage_golden/dump_mirage_golden.cpp -o /tmp/dump_mirage_golden
   /tmp/dump_mirage_golden > lib/segment/src/index/mirage_index/tests/fixtures/mirage_cpp_golden_n64_d8_l2.json
   ```

   如果 dumper 已改为链接真实 FAISS/Mirage 实现，编译命令需要替换为对应 C++ reference 工程的构建命令。文档中的 `clang++` 单文件命令只适用于当前自包含 dumper，不足以生成 authoritative `layer0_injected`。

4. 扩展 Rust fixture struct：

   ```rust
   pub hnsw_m0: usize
   pub layer0_top_s: Vec<Vec<usize>>
   pub layer0_injected: Vec<Vec<usize>>
   pub layer0_injected_stats: Layer0Stats
   ```

5. 先加低层 exact test：
   - `LinksContainer::fill_from_sorted_with_heuristic`
   - 对齐真实 C++ `init_level_0_from_knngraph` dump 出来的 `layer0_injected`

6. 再加 builder exact test：
   - `GraphLayersBuilder::inject_layer0_with_heuristic`

7. 最后加真实 build-path test：
   - `MirageIndex::build`
   - `GraphLayers::load`
   - 读取 level 0 links
   - 对比 `layer0_injected`

## 8. 验收标准

最低验收：

```text
Rust fill_from_sorted_with_heuristic output == C++ init_level_0_from_knngraph 后的 Layer 0 output
Rust inject_layer0_with_heuristic output == C++ init_level_0_from_knngraph Layer 0 output
```

更高一级验收：

```text
MirageIndex::build persisted GraphLayers Layer 0 == C++ init_level_0_from_knngraph Layer 0 output
```

统计验收：

```text
degree histogram exact match
total_edges exact match
reciprocity ratio exact match
weak connectivity exact match
```

如果 exact 不通过，先不要放宽阈值。应该先定位：

- C++ priority_queue 顺序是否一致
- fixture.layer0_injected 是否确实来自真实 hierarchy.hnsw.neighbors，而不是模拟 shrink
- Rust candidate 输入是否已经 sorted closest-first
- score/distance 方向是否反了
- 是否只取了 top S
- `m0` 是否和 C++ `dest_size` 一致
- 是否存在浮点距离 tie

## 9. 工程上最重要的一点

这一步验证的不是 Mirage refinement 本体，那个已经有 golden test 了。

这一步验证的是：

```text
Mirage refinement output
进入 HNSW-compatible Layer 0 graph storage 前后
有没有发生语义偏移
```

做完以后，C++ 对齐证据链会变成：

```text
C++ Mirage Layer 0 refinement
== Rust Mirage Layer 0 refinement

C++ HNSW level assignment / shuffle
== Rust FAISS-compatible level assignment / shuffle

C++ init_level_0_from_knngraph
== Rust inject_layer0_with_heuristic

Rust MirageIndex::build persisted GraphLayers
== expected injected Layer 0 graph
```

这样才可以更有底气地说：Mirage 在 Qdrant 中不仅“算法前半段像 C++”，而是“从 Layer 0 生成到 Qdrant HNSW-compatible 图注入这一段都已经被 C++ golden parity 覆盖”。
