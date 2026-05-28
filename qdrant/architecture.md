# Qdrant Mirage Index Architecture

Date: 2026-05-27

## 1. Scope

This document summarizes the Mirage index integration that currently exists in this Qdrant fork.

The implementation is not a minimal dummy backend. It is a Rust-native Mirage-style graph builder that produces an HNSW-compatible graph and then reuses Qdrant's existing HNSW runtime for search, file handling, telemetry, and cache operations.

The current implementation should be viewed as an experimental dense vector index path. The main missing piece is a clean trigger path from normal collection optimization into `Indexes::Mirage`.

## 2. Product Context

Qdrant is positioned as a production-ready vector search engine and vector database. The repository README emphasizes:

- Production-ready point storage and vector similarity search.
- Extended payload filtering.
- Vector quantization and on-disk storage.
- Distributed deployment via sharding and replication.
- Stable REST and gRPC APIs.

That product context matters for Mirage integration. A production-quality index cannot be judged only by whether it returns nearest-neighbor results. It must eventually interact correctly with filtering, optimizer lifecycle, snapshots, quantization, persistence, update/delete semantics, and API compatibility.

The current implementation makes a pragmatic architecture choice: reuse HNSW runtime behavior where possible, and limit new code mostly to graph construction.

## 3. Core Design Decision

Mirage and HNSW have compatible hierarchical graph shapes:

- Layer 0 contains all indexed points.
- Upper layers contain geometrically decreasing subsets.
- Search can use standard HNSW top-down greedy / beam traversal.

The main algorithmic difference is construction:

- HNSW builds Layer 0 incrementally through search-and-connect.
- Mirage builds Layer 0 through refinement:
  - initialize a random `S`-regular graph,
  - run `R` rounds of local RNG-style pruning,
  - run `iter` updates per round,
  - re-inject reverse edges between rounds,
  - cap reverse-edge expansion by `num_reverse_edges`.

The implementation therefore does not introduce a separate query engine. It builds a graph in Mirage style and opens it as an HNSW-compatible graph.

## 4. Implemented Module Map

Main Mirage module:

- `lib/segment/src/index/mirage_index/mod.rs`
  - Declares the Mirage module.
  - Re-exports `MirageIndex`, `MirageIndexOpenArgs`, `MirageGraphConfig`, and `RefinementParams`.

- `lib/segment/src/index/mirage_index/config.rs`
  - Defines persisted `MirageGraphConfig`.
  - Writes `mirage_config.json`.
  - Converts Mirage graph config into HNSW-compatible `HnswGraphConfig`.

- `lib/segment/src/index/mirage_index/refinement_builder.rs`
  - Implements the Rust-native Layer 0 refinement algorithm.
  - Uses Qdrant's score convention: higher score means closer.
  - Contains a synthetic Layer 0 recall sanity test.

- `lib/segment/src/index/mirage_index/mirage.rs`
  - Defines `MirageIndex`.
  - Builds Mirage Layer 0.
  - Builds upper layers through HNSW builder methods.
  - Persists a standard HNSW-compatible graph.
  - Opens an inner `HNSWIndex` and delegates runtime search.

HNSW builder extensions used by Mirage:

- `lib/segment/src/index/hnsw_index/graph_layers_builder.rs`
  - `link_new_point_with_min_level(...)`
    - Allows Mirage to build only layers `>= 1` using standard HNSW logic.
  - `inject_layer0_with_heuristic(...)`
    - Allows Mirage to inject precomputed Layer 0 candidates and apply the same HNSW RNG-style heuristic pruning.

Index abstraction integration:

- `lib/segment/src/types.rs`
  - Adds `Indexes::Mirage(MirageConfig)`.
  - Defines `MirageConfig` and defaults:
    - `m = 16`
    - `ef_construct = 1024`
    - `s = 32`
    - `r = 4`
    - `iter = 15`
    - `num_reverse_edges = 96`
  - Implements Mirage mismatch comparison.

- `lib/segment/src/index/vector_index_base.rs`
  - Adds `VectorIndexEnum::Mirage(MirageIndex)`.
  - Wires Mirage into:
    - `search`
    - `get_telemetry_data`
    - `files`
    - `immutable_files`
    - `indexed_vector_count`
    - `size_of_searchable_vectors_in_bytes`
    - `update_vector`
    - cache/populate helpers.

- `lib/segment/src/segment_constructor/segment_constructor_base.rs`
  - Opens and builds `MirageIndex` when segment config contains `Indexes::Mirage`.

Collection/shard compatibility hooks:

- `lib/collection/src/collection_manager/segments_searcher.rs`
  - Treats Mirage `ef_construct` like HNSW for dynamic ef selection.

- `lib/shard/src/optimizers/config_mismatch_optimizer.rs`
  - Handles `Indexes::Mirage` in mismatch checks by converting Mirage config to an HNSW-compatible view.
  - Current limitation: Mirage-specific knobs are not yet first-class optimizer config inputs.

- `lib/edge/src/config/vectors.rs`, `lib/edge/src/config/shard.rs`, and `lib/edge/python/src/config/vector_data.rs`
  - Expose Mirage through an HNSW-compatible view where needed.

## 5. Configuration Model

### User-Facing Config

`MirageConfig` lives in `lib/segment/src/types.rs`.

It contains HNSW-compatible fields:

- `m`
- `ef_construct`
- `full_scan_threshold`
- `max_indexing_threads`
- `on_disk`
- `payload_m`

It also contains Mirage-specific Layer 0 refinement fields:

- `s`
- `r`
- `iter`
- `num_reverse_edges`

`MirageConfig::to_hnsw_compat()` converts the overlapping fields into `HnswConfig`.

### Persisted Graph Config

`MirageGraphConfig` lives in `lib/segment/src/index/mirage_index/config.rs`.

It persists:

- HNSW-compatible graph shape:
  - `m`
  - `m0`
  - `ef_construct`
  - `ef`
  - `full_scan_threshold`
  - `payload_m`
  - `payload_m0`
  - `indexed_vector_count`
- Mirage build parameters:
  - `s`
  - `r`
  - `iter`
  - `num_reverse_edges`
- build threading:
  - `max_indexing_threads`

The file name is:

```text
mirage_config.json
```

The implementation also saves an HNSW-compatible config next to the graph so `HNSWIndex::open` can load the graph normally.

## 6. Build Flow

The entry point is `MirageIndex::build(...)`.

High-level flow:

1. Check that no Mirage/HNSW graph files already exist at the target path.
2. Create the index directory.
3. Borrow:
   - `IdTrackerEnum`
   - `VectorStorageEnum`
   - optional `QuantizedVectors`
4. Convert `full_scan_threshold` from KB semantics into vector-count semantics.
5. Build a `MirageGraphConfig`.
6. Create a Rayon thread pool using the build permit CPU count.
7. Build the graph with `build_main_graph(...)`.
8. Persist graph layers using the standard HNSW graph format.
9. Save:
   - HNSW-compatible graph config,
   - Mirage-specific graph config.
10. Open an inner `HNSWIndex` over the generated graph.
11. Return `MirageIndex { inner, config, path }`.

### Layer 0 Build

`build_main_graph(...)` is the core Mirage construction path.

It:

1. Iterates all live internal point IDs from the id tracker.
2. Assigns each live point a random HNSW level using the existing HNSW level distribution.
3. Builds Mirage Layer 0 adjacency through `refinement_builder::build_layer0(...)`.
4. Injects each Layer 0 adjacency list into `GraphLayersBuilder` with `inject_layer0_with_heuristic(...)`.

The Layer 0 refinement builder:

1. Builds a liveness bitset.
2. Initializes each live point with `S` random live neighbors.
3. Runs `R` rounds.
4. Each round performs `iter` update passes.
5. Between rounds, reverse-edge candidates are merged back into each vertex pool, capped by `num_reverse_edges`.
6. Final output is sorted closest-first and deduplicated.

The implementation uses Qdrant's score convention. Higher score means closer. For distance-style comparisons from the Mirage paper, the pruning condition is inverted accordingly.

### Upper Layer Build

Upper layers are built using HNSW logic.

Mirage calls:

```text
GraphLayersBuilder::link_new_point_with_min_level(point_id, scorer, 1)
```

This means:

- Layer 0 remains the Mirage-refined layer.
- Layers 1..N are built with Qdrant's existing HNSW search-and-connect behavior.

## 7. Runtime Search Flow

`MirageIndex` implements `VectorIndex`.

At runtime:

```text
MirageIndex::search(...)
  -> self.inner.search(...)
```

Search is delegated directly to the inner `HNSWIndex`.

This gives Mirage the same runtime search machinery as HNSW:

- unfiltered graph traversal,
- filtered planning behavior through HNSW,
- plain-scan fallback logic,
- query context and hardware counters,
- quantization-aware runtime paths where HNSW supports them,
- telemetry collection.

Telemetry changes only the index label:

```text
index_name = "mirage"
```

## 8. Persistence and Files

The current implementation is not purely in-memory.

During build it writes standard HNSW-compatible graph files and config, then reopens them through `HNSWIndex`.

`MirageIndex::files()` returns:

- inner HNSW files,
- `mirage_config.json` if it exists.

`immutable_files()` returns the same file set.

This is more advanced than a strict P0 in-memory prototype, but it also means persistence and snapshot behavior must be tested carefully before claiming production readiness.

## 9. Current Behavior Summary

Implemented:

- Rust-native Mirage Layer 0 refinement builder.
- Mirage config structs and defaults.
- `Indexes::Mirage`.
- `VectorIndexEnum::Mirage`.
- Segment constructor open/build support when segment config already contains Mirage.
- HNSW-compatible graph persistence.
- HNSW runtime delegation for search.
- Telemetry label override.
- File listing that includes `mirage_config.json`.
- HNSW graph builder hooks needed by Mirage.
- Basic collection/shard compatibility handling for `ef_construct` and config mismatch.

Not yet fully implemented or not yet proven:

- Public collection API for selecting Mirage.
- Hidden environment-variable override for forcing Mirage.
- Optimizer generation of `Indexes::Mirage`.
- Mirage-specific optimizer config model.
- Explicit P0 guardrails for no filter / no quantization / no payload subgraph.
- Segment-level and REST-level smoke tests proving search requests enter Mirage.
- Benchmark suite comparing build time, recall, and query latency against HNSW and exact search.
- Production-grade support matrix for payload filtering, quantization, multi-vector, snapshots, updates/deletes, and distributed deployment.

## 10. Important Current Gap

The main architecture gap is triggerability.

`build_vector_index(...)` and `open_vector_index(...)` can instantiate Mirage if `VectorDataConfig.index` is already `Indexes::Mirage`.

However, the optimizer path that turns a large plain dense segment into an indexed dense segment still assigns HNSW:

```text
config.index = Indexes::Hnsw(vector_cfg.hnsw_config)
```

Therefore a normal collection using standard HNSW config will not naturally build Mirage yet.

The next engineering step should be to add an experimental trigger path that can produce `Indexes::Mirage` during segment optimization without changing the public API first.

## 11. Engineering Assessment

The current approach is stronger than a C++ FFI dummy prototype for long-term integration:

- It avoids cross-language build and deployment complexity.
- It fits Qdrant's Rust-native codebase.
- It reuses HNSW runtime behavior instead of duplicating search logic.
- It keeps Mirage's algorithmic surface concentrated in graph construction.

The cost is that the implementation jumped past the smallest possible P0. It now needs a small, strict validation loop to prove that the existing deeper implementation works through Qdrant's normal search path.
