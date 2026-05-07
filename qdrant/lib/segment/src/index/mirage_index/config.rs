//! On-disk graph configuration for the MIRAGE index.
//!
//! This is the analog of [`crate::index::hnsw_index::config::HnswGraphConfig`],
//! but persists the additional MIRAGE-specific Layer 0 refinement parameters.
//!
//! The MIRAGE index produces a graph that is *structurally* identical to an
//! HNSW graph (same level system, same on-disk `graph.bin`/`links.bin` files).
//! The only difference is how Layer 0 is constructed:
//!
//! - HNSW builds Layer 0 incrementally via greedy beam search + RNG-style
//!   pruning, one point at a time.
//! - MIRAGE builds Layer 0 via a refinement pass: start with a random
//!   `S`-regular graph, run `R` rounds × `iter` updates of local RNG-rule
//!   pruning, with reverse-edge re-injection between rounds.
//!
//! Layers 1..N are constructed by both algorithms in the same way (HNSW's
//! standard top-down ANNS-driven insertion).
//!
//! Persisting the MIRAGE construction parameters is required so that we
//! detect configuration mismatches that require a rebuild.

use std::path::{Path, PathBuf};

use common::fs::{atomic_save_json, read_json};
use serde::{Deserialize, Serialize};

use crate::common::operation_error::OperationResult;

/// File name for the persisted [`MirageGraphConfig`].
pub const MIRAGE_INDEX_CONFIG_FILE: &str = "mirage_config.json";

/// Persisted graph configuration for the MIRAGE index.
///
/// This struct is written to `mirage_config.json` next to the standard HNSW
/// `graph.bin` / `links.bin` files. It includes both the HNSW-equivalent
/// parameters (so we can reconstruct an [`HnswGraphConfig`] view for the
/// shared search and storage code paths) and the MIRAGE-specific Layer 0
/// refinement parameters.
///
/// [`HnswGraphConfig`]: crate::index::hnsw_index::config::HnswGraphConfig
#[derive(Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq)]
pub struct MirageGraphConfig {
    /// `M` for upper layers (Layers 1..N).
    pub m: usize,

    /// Effective `m0` for Layer 0. The MIRAGE refinement may produce up to
    /// `num_reverse_edges` neighbors per vertex, but the on-disk Layer 0
    /// store is capped to `m0` slots (RNG-pruned during injection).
    pub m0: usize,

    /// `ef_construct` used when building upper layers (Layers ≥ 1).
    ///
    /// The reference MIRAGE C++ implementation hardcodes this to 1024,
    /// significantly larger than typical HNSW (Qdrant default 100). The
    /// upper layers being thoroughly built is one of the contributors to
    /// MIRAGE's superior search QPS at fixed recall.
    pub ef_construct: usize,

    /// `ef` used at search time. Equals `ef_construct` by default but can
    /// be overridden via search params.
    pub ef: usize,

    /// Threshold (in number of vectors) below which we prefer plain scan.
    pub full_scan_threshold: usize,

    /// Initial out-degree of the random graph at Layer 0 (paper's `S`).
    /// Recommended: 32.
    pub s: usize,

    /// Number of refinement rounds at Layer 0 (paper's `R`).
    /// Recommended: 4.
    pub r: usize,

    /// Number of NN-Descent / RNG-pruning iterations within each refinement
    /// round (paper's `Iter`). Recommended: 12–15.
    pub iter: usize,

    /// Maximum number of neighbors per vertex retained when merging reverse
    /// edges back into Layer 0. The reference C++ implementation hardcodes
    /// this to 96.
    pub num_reverse_edges: usize,

    /// Number of parallel threads used during background index building.
    /// 0 means autodetect.
    #[serde(default)]
    pub max_indexing_threads: usize,

    /// Optional `payload_m` for filterable payload sub-graphs (forwarded to
    /// HNSW upper-layer plumbing). Phase 1 doesn't build payload sub-graphs,
    /// but the field is preserved for forward compatibility.
    #[serde(default)]
    pub payload_m: Option<usize>,

    /// `payload_m0` derived from `payload_m`.
    #[serde(default)]
    pub payload_m0: Option<usize>,

    /// Number of indexed vectors at the time of the last build. Used to
    /// detect when an optimizer rebuild is needed.
    #[serde(default)]
    pub indexed_vector_count: Option<usize>,
}

impl MirageGraphConfig {
    /// Build a fresh config from user-provided parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        ef_construct: usize,
        full_scan_threshold: usize,
        s: usize,
        r: usize,
        iter: usize,
        num_reverse_edges: usize,
        max_indexing_threads: usize,
        payload_m: Option<usize>,
        indexed_vector_count: usize,
    ) -> Self {
        // Mirror HNSW: m0 = 2 * m by default. The user-facing MirageConfig
        // doesn't expose m0 separately; fix it here.
        let m0 = m.saturating_mul(2);
        MirageGraphConfig {
            m,
            m0,
            ef_construct,
            ef: ef_construct,
            full_scan_threshold,
            s,
            r,
            iter,
            num_reverse_edges,
            max_indexing_threads,
            payload_m,
            payload_m0: payload_m.map(|v| v.saturating_mul(2)),
            indexed_vector_count: Some(indexed_vector_count),
        }
    }

    pub fn get_config_path(path: &Path) -> PathBuf {
        path.join(MIRAGE_INDEX_CONFIG_FILE)
    }

    pub fn load(path: &Path) -> OperationResult<Self> {
        Ok(read_json(path)?)
    }

    pub fn save(&self, path: &Path) -> OperationResult<()> {
        Ok(atomic_save_json(path, self)?)
    }

    /// Build a HNSW-compatible view of this config so that the existing
    /// HNSW search and storage code paths can be reused unmodified.
    ///
    /// The returned [`HnswGraphConfig`] has:
    /// - `m`, `m0`, `ef_construct`, `ef` taken straight from this config,
    /// - `full_scan_threshold` taken straight,
    /// - `payload_m`/`payload_m0` taken straight,
    /// - `indexed_vector_count` taken straight,
    /// - `max_indexing_threads` taken straight.
    ///
    /// This view is *only* meant for runtime search and HNSW-side bookkeeping;
    /// MIRAGE-specific build params (`s`, `r`, `iter`, `num_reverse_edges`)
    /// are not exposed by it because they only matter at build time.
    ///
    /// [`HnswGraphConfig`]: crate::index::hnsw_index::config::HnswGraphConfig
    pub fn to_hnsw_compat(&self) -> crate::index::hnsw_index::config::HnswGraphConfig {
        let mut cfg = crate::index::hnsw_index::config::HnswGraphConfig {
            m: self.m,
            m0: self.m0,
            ef_construct: self.ef_construct,
            ef: self.ef,
            full_scan_threshold: self.full_scan_threshold,
            max_indexing_threads: self.max_indexing_threads,
            payload_m: self.payload_m,
            payload_m0: self.payload_m0,
            indexed_vector_count: self.indexed_vector_count,
        };
        // Defensive: if `m == 0` the user disabled the upper layers, which
        // for MIRAGE means we still have a Layer-0-only graph. Mirror what
        // HNSW does in that mode.
        if cfg.m == 0 {
            cfg.m0 = 0;
        }
        cfg
    }
}
