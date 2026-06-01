# Mirage 实现策略修订

日期：2026-05-29

## 1. 背景

`0528_plan.md` 的目标是把 Mirage 从“手动配置可用”推进到“实验开关可触发，并能被测试验证”。后续 code review 发现当前 Rust/Qdrant 版本存在一个关键阻断项：

- `test_layer0_refinement_recall` 中 Layer 0 邻接表 recall@10 只有约 `0.445`。
- 该 recall 不能满足 `0.9+` 的验收要求。
- 当前 Rust 实现还没有先严格对齐本地 Mirage C++ 源码，因此不能通过自行修改算法来“救 recall”。

本修订方案的原则是：

```text
先按 Mirage C++ 源码做语义等价移植，再用 Qdrant segment-level search recall 验收。
```

Mirage 原理不能改变。任何提升 recall 的改动都必须能在原 C++ 实现中找到对应语义，或明确标注为 Qdrant 适配层行为。

## 2. 源码基准

后续实现以本地 Mirage 源码为准：

- `/Users/krystal/Desktop/入职材料/Mirage 复现/mirage/faiss/impl/MIRAGE.cpp`
- `/Users/krystal/Desktop/入职材料/Mirage 复现/mirage/faiss/impl/MIRAGE.h`
- `/Users/krystal/Desktop/入职材料/Mirage 复现/mirage/faiss/IndexMIRAGE.cpp`
- `/Users/krystal/Desktop/入职材料/Mirage 复现/mirage/faiss/IndexHNSW.cpp`
- `/Users/krystal/Desktop/入职材料/Mirage 复现/mirage/faiss/impl/HNSW.cpp`

当前 Qdrant 侧重点文件：

- `lib/segment/src/index/mirage_index/refinement_builder.rs`
- `lib/segment/src/index/mirage_index/mirage.rs`
- `lib/segment/src/types.rs`
- `lib/segment/src/index/hnsw_index/graph_layers_builder.rs`

## 3. 总体目标

实现分两层：

1. 算法层严格对齐 C++ Mirage。
   - `init_graph`
   - `update`
   - `add_reverse_edges`
   - Layer 0 转 HNSW graph 的输入语义
   - 默认参数

2. Qdrant 适配层只做必要转换。
   - distance/score 方向转换
   - live/deleted point 过滤
   - Qdrant graph 持久化
   - segment-level build/open/search 测试

验收标准从“Layer 0 adjacency recall”调整为：

```text
Mirage segment no-filter search recall@10 >= 0.9
```

Layer 0 adjacency recall 不作为产品 recall 指标。

## 4. 阶段一：参数对齐

C++ 默认参数在 `MIRAGE.h` 中：

```cpp
int R = 4;
int iter = 15;
int S = 16;
int random_seed = 2021;
int L = 8;
```

Rust 当前默认 `S=32`，应先对齐 C++。

改动：

1. `lib/segment/src/types.rs`
   - `DEFAULT_MIRAGE_S` 改为 `16`。
   - `DEFAULT_MIRAGE_R` 保持 `4`。
   - `DEFAULT_MIRAGE_ITER` 保持 `15`。
   - `DEFAULT_MIRAGE_NUM_REVERSE_EDGES` 保持 `96`。

2. `lib/segment/src/index/mirage_index/refinement_builder.rs`
   - `RefinementParams::default().s` 改为 `16`。
   - 注释从“paper/README default”改为“matches local C++ Mirage implementation”。

3. `MirageConfig::from_hnsw(...)`
   - `ef_construct.max(1024)` 保留。
   - 原因：C++ `IndexMIRAGE.cpp::add_levels()` 显式设置 `hnsw.efConstruction = 1024`。

## 5. 阶段二：重写 init_graph

C++ 语义：

```cpp
gen_random(rng, tmp.data(), S, ntotal);
for (int j = 0; j < S; j++) {
    int id = tmp[j];
    if (id == i) continue;
    float dist = qdis.symmetric_dis(i, id);
    graph[i].pool.push_back(Neighbor(id, dist, true));
}
std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
graph[i].pool.reserve(L);
```

当前 Rust 使用 `choose_multiple` 从 alive set 中排除自己后采样，这和 C++ 不等价。

改动：

1. 在 `refinement_builder.rs` 增加 C++ 等价随机生成函数：

```rust
fn gen_random_cpp_style(rng: &mut StdRng, size: usize, n: usize) -> Vec<usize>
```

行为对齐：

- 生成 `size` 个范围内随机数。
- 排序。
- 对重复/递减位置做 `addr[i] = addr[i - 1] + 1`。
- 加随机 offset 后取模。

2. `init_random_graph(...)` 改为：

- 先按 C++ 方式生成 `S` 个候选。
- 如果候选是 self，跳过。
- 如果候选不是 alive，跳过。
- 插入 `Candidate { idx, score, new_flag: true }`。

3. 保持 Qdrant score 语义：

- C++ distance 越小越近。
- Qdrant score 越大越近。
- Rust 排序仍使用 score descending。

## 6. 阶段三：重写 update_round

C++ `Mirage::update(...)` 关键语义：

1. 对每个点 `u`：
   - lock 当前 pool。
   - `old_pool.swap(nhood.pool)`。
2. 对 `old_pool`：
   - sort。
   - unique by id。
3. 遍历 candidate：
   - 如果 candidate 已经进入 `new_ids`，跳过。
   - 和 `new_pool` 中已有点做 RNG 检查。
   - 如果 C++ `distance(nn.id, other_nn.id) < nn.distance`，则 reject，并 `insert_nn(other_nn.id, nn.id, distance, true)`。
4. 对保留的 candidate：
   - `flag = false`。
5. 直接写回：
   - `nhood.pool = std::move(new_pool)`。

当前 Rust 在写回时会把并发插入到 `p.inner` 的内容 merge 回来，这不是 C++ 行为。

改动：

1. `update_round(...)` 写回时改为直接覆盖：

```rust
let mut p = pools[u].lock();
p.inner = new_pool;
```

2. 不再保留 concurrent reverse-edge additions。

3. RNG 条件保持 score 等价转换：

```rust
// C++: distance(nn, other) < distance(u, nn)
// Rust: score(nn, other) > score(u, nn)
if score_nn_other > nn.score {
    insert into other pool
}
```

4. 保留 `old/old` flag skip 逻辑，因为 C++ 有：

```cpp
if (!nn.flag && !other_nn.flag)
    continue;
```

## 7. 阶段四：重写 add_reverse_edges

C++ `Mirage::add_reverse_edges()` 语义不是简单把 reverse pool merge 回当前点，而是：

1. 构造 `reverse_pools[ntotal]`。
2. 对每条边 `u -> nn.id`：
   - `reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag)`。
3. 对每个点 `u`：
   - 将 `graph[u].pool` 中所有 flag 置为 true。
   - 将当前 pool append 到 `reverse_pools[u]`。
   - sort。
   - unique by id。
   - truncate 到 `num_reverse_edges = 96`。
4. 回灌：
   - 对 `reverse_pools[u]` 中每个 `nn`：
   - `graph[nn.id].pool.emplace_back(u, nn.distance, nn.flag)`。
5. 最终对每个 `graph[u].pool`：
   - sort。
   - truncate 到 96。

当前 Rust 只做了“reverse merge 到当前 pool”，少了 C++ 的回灌顺序和中间截断语义。

改动：

在 `refinement_builder.rs` 将 `add_reverse_edges(...)` 拆成四个阶段：

1. scatter：

```rust
reverse[nn.idx].push(Candidate { idx: u, score: nn.score, new_flag: nn.new_flag })
```

2. local combine：

```rust
for c in current_pool {
    c.new_flag = true;
}
reverse[u].append(current_pool);
sort/dedup/truncate(num_reverse_edges);
clear current pool;
```

3. backfill：

```rust
for nn in reverse[u] {
    pools[nn.idx].push(Candidate { idx: u, score: nn.score, new_flag: nn.new_flag });
}
```

4. final normalize：

```rust
sort/dedup/truncate(num_reverse_edges)
```

注意：

- 保持 score descending 排序。
- dedup by `idx`。
- 不增加 C++ 没有的 reciprocal repair 或 per-iter reverse。

## 8. 阶段五：Layer 0 注入 HNSW 时只取前 S 个

C++ `IndexMIRAGE::convert_to_mirage(...)`：

```cpp
int k = mirage.S;
...
for (int i = 0; i < degree && i < k; ++i) {
    int neighbor_id = mirage.final_graph[offset + i];
    I[u * k + i] = neighbor_id;
    D[u * k + i] = (*local_qdis)(neighbor_id);
}
hierarchy.init_level_0_from_knngraph(k, D.data(), I.data());
```

也就是说，C++ 转 HNSW Layer 0 时每个点最多取 `S` 个候选。

当前 Rust 将完整 `layer0_adj[u]` 传入 `inject_layer0_with_heuristic(...)`。

改动：

在 `lib/segment/src/index/mirage_index/mirage.rs`：

```rust
builder.inject_layer0_with_heuristic(
    pid,
    nbrs.iter().copied().take(config.s),
    |a, b| scorer.score_internal(a, b),
);
```

这更接近 C++ 的 `init_level_0_from_knngraph(k = S, ...)`。

## 9. 阶段六：上层 HNSW 构建对齐检查

C++ 上层构建：

1. `prepare_level_tab_mirage(n, false)` 为所有点生成 levels。
2. 按 level bucket。
3. 从最高 level 到 1 逐层插入。
4. 同层随机打乱顺序。
5. 调用 `hnsw.add_with_locks(...)`。

当前 Rust：

- 先为所有 live point 设置 random level。
- 注入 Layer 0。
- 再对所有 alive point 调 `link_new_point_with_min_level(pid, ..., 1)`。

这与 C++ 不完全一致，但涉及 Qdrant HNSW builder 内部 API，建议分两步：

1. 第一轮先完成 Layer 0 对齐。
2. 如果 segment search recall 仍达不到 `0.9+`，再实现 Qdrant 版 level bucket build：
   - 按 `builder.get_point_level(pid)` 分桶。
   - 从最高 level 到 1 插入。
   - 同层用 deterministic seed shuffle。
   - 调用等价的 min-level link API。

## 10. 阶段七：测试策略

### 10.1 refinement_builder 结构测试

不再要求 Layer 0 adjacency recall@10 >= 0.9。

Layer 0 adjacency 是 RNG-pruned graph，不等价于 exact KNN top10 list。

保留/新增：

- 无 self-loop。
- 无重复边。
- score descending。
- `add_reverse_edges` 后每点 degree 不超过 `num_reverse_edges`。
- empty segment / only one alive point。
- deleted point 不被采样为邻居。

### 10.2 C++ 语义对齐测试

小规模 deterministic 测试：

- 固定 N、D、seed。
- 跑 Rust `build_layer0`。
- 记录：
  - 平均 degree。
  - max degree。
  - min degree。
  - pool 分布。
  - 前 S 个候选的稳定性。

如果条件允许，新增一个脚本跑本地 C++ Mirage 小数据输出同样统计，用于人工对照。

### 10.3 segment-level search recall 测试

这是核心验收。

测试要求：

- 构建 1000 到 5000 条向量。
- 维度 32 或 64。
- deterministic seed。
- 配置 `Indexes::Mirage`。
- 触发 index build。
- 断言实际索引是 `VectorIndexEnum::Mirage`。
- 执行 no-filter search。
- exact/plain search 作为 ground truth。
- `recall@10 >= 0.9`。

### 10.4 small-N boundary 与 benchmark 分组

`N <= S` uses Qdrant-specific complete graph initialization to avoid the C++ `gen_random(N-S)` invalid boundary. These runs are correctness-only and must not be included in Mirage C++ parity or recall@10 acceptance metrics.

报告必须拆成两类：

```text
Mirage C++ parity benchmark:
  cpp_parity_eligible=true
  N > S
  N >= 10 * S for recall acceptance
  report recall/build

Qdrant boundary behavior:
  cpp_parity_eligible=false
  N <= S
  boundary_behavior=qdrant_complete_graph
  metric_scope=correctness_only
  do not compare with C++
```

正式 recall 验收不得使用 tiny dataset：

- `N <= S`：只做 Qdrant boundary correctness，不参与 C++ parity。
- `S < N < 10*S`：C++ random-init path 可运行，但不得输出/断言正式 recall 结论。
- 推荐正式 benchmark 使用 `N >= max(100*S, 10000)`。

### 10.5 persistence/open 测试

测试：

- build Mirage segment。
- drop segment。
- reopen segment。
- 确认 `mirage_config.json` 和 HNSW-compatible graph 文件存在。
- reopen 后 no-filter search recall 不下降。

## 11. 阶段八：发布门禁

在以下条件满足前，不允许把 Mirage 作为自动 optimizer 默认产物：

- Layer 0 构建语义已和 C++ 对齐。
- segment-level no-filter search recall@10 >= 0.9。
- build/open/search 集成测试通过。
- mismatch optimizer 不会导致重复 rebuild。
- P0 guard 覆盖 dense Float32/no quantization/no mmap/no filter/no multivector。

在此之前：

- `QDRANT_EXPERIMENTAL_MIRAGE=1` 最多作为手动实验入口。
- 不声称支持 filter。
- 不声称支持 quantization。
- 不声称支持 mmap/on-disk vector storage。
- 不声称支持 multi-vector。

## 12. 推荐实施顺序

1. 参数默认值对齐 C++。
2. `init_random_graph` 对齐 C++ `gen_random + init_graph`。
3. `update_round` 对齐 C++ `Mirage::update`。
4. `add_reverse_edges` 对齐 C++ `Mirage::add_reverse_edges`。
5. Layer 0 注入改为 `.take(config.s)`。
6. 修正/替换当前 Layer 0 recall 单测。
7. 新增 segment-level no-filter search recall@10 测试。
8. 跑 `cargo test -p segment mirage --lib`。
9. 跑新增 integration test。
10. 如 recall 仍不足，再处理 upper-layer build 顺序对齐。

## 13. 结论

当前 recall 问题不能通过随意调参或改 Mirage 原理解决。

正确路径是：

```text
严格对齐 /mirage/faiss/impl/MIRAGE.cpp 的 Layer 0 构建语义
-> 对齐 Layer 0 转 HNSW 的输入语义
-> 用 Qdrant segment search recall@10 >= 0.9 验收
```

只有通过该验收后，Mirage 才能从“架构接入”推进到“可用 MVP”。
