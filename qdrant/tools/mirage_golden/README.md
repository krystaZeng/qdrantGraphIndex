# MIRAGE Golden Fixtures

This directory contains C++ fixture dumpers for MIRAGE parity tests.

## Authoritative Dumper

`dump_mirage_authoritative.cpp` is the preferred source for checked-in golden
fixtures. It must be built against the local FAISS/MIRAGE C++ reference
implementation and dumps `layer0_injected` from the real
`hierarchy.hnsw.neighbors` table after calling:

```cpp
hierarchy.init_level_0_from_knngraph(k, D.data(), I.data());
```

Use this dumper whenever the fixture is used to claim C++ Layer-0 injection
parity.

The exact build command depends on the local FAISS build setup and its
OpenMP/BLAS dependencies. A typical workflow is:

```bash
# Build/link against the local HNSW_level0_compute_distance/faiss sources.
# Then redirect the output into:
lib/segment/src/index/mirage_index/tests/fixtures/mirage_cpp_golden_n64_d8_l2.json
```

## Diagnostic Dumper

`dump_mirage_golden.cpp` is self-contained and mirrors the local reference
implementation in:

- `faiss/impl/MIRAGE.cpp`
- `faiss/impl/HNSW.cpp`
- `faiss/utils/random.cpp`

It intentionally does not link FAISS. It is useful for deterministic debugging,
but its `layer0_injected` field is produced by a self-contained diagnostic
shrink implementation. Do not use that field as the authoritative source for a
C++ parity claim; prefer `dump_mirage_authoritative.cpp`.

Regenerate the diagnostic fixture from the repository root:

```bash
clang++ -std=c++17 -O2 tools/mirage_golden/dump_mirage_golden.cpp -o /tmp/dump_mirage_golden
/tmp/dump_mirage_golden > lib/segment/src/index/mirage_index/tests/fixtures/mirage_cpp_golden_n64_d8_l2.json
```

Keep both dumpers single-threaded. Golden tests are intended to verify
algorithmic parity, not parallel scheduling behavior.
