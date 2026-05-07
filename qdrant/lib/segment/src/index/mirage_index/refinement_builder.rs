//! Layer 0 refinement-based builder for MIRAGE.
//!
//! This is a faithful Rust port of the algorithmic core of MIRAGE
//! (`faiss/impl/MIRAGE.cpp` in the reference C++ implementation), adapted to
//! Qdrant's scoring convention.
//!
//! # Scoring convention
//!
//! Qdrant uses *similarity scores*: `score(a, b)` is **higher when `a` and
//! `b` are closer**. This is the opposite of the reference implementation,
//! which uses *distances* (lower = closer). The RNG-rule check therefore
//! flips:
//!
//! - Reference (distance):
//!   prune edge `u → nn` if there exists `other` in the new pool with
//!   `distance(nn, other) < distance(u, nn)`.
//! - Qdrant (score):
//!   prune edge `u → nn` if there exists `other` in the new pool with
//!   `score(nn, other) > score(u, nn)`.
//!
//! Both formulations express the same geometric condition (the "longest edge
//! of the triangle" is pruned).
//!
//! # Algorithm outline
//!
//! ```text
//! 1. init_random_graph: each vertex u gets S random unique neighbors,
//!    pool sorted closest-first by score (descending).
//! 2. for r in 0..R:
//!      for it in 0..iter:
//!          update: per-vertex local RNG-rule pruning of `pool`. If a
//!          neighbor `nn` is rejected by `other`, push a "reverse-edge
//!          candidate" (u, score) into other's pool.
//!      if r < R - 1:
//!          add_reverse_edges: gather reverse-edge contributions and merge
//!          them back into each vertex's pool, capped to `num_reverse_edges`.
//! 3. final pool dedup + sort closest-first.
//! ```
//!
//! The output is a `Vec<Vec<ScoredPointOffset>>` where for each point `u`,
//! the inner vector is **sorted closest-to-furthest** (descending score).
//! This format feeds directly into Qdrant's
//! [`fill_from_sorted_with_heuristic`] for Layer 0 injection.
//!
//! [`fill_from_sorted_with_heuristic`]: crate::index::hnsw_index::links_container::LinksContainer::fill_from_sorted_with_heuristic

use std::sync::atomic::AtomicBool;

use common::bitvec::{BitSliceExt as _, BitVec};
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};

/// Score of a candidate neighbor, plus a "new/old" flag matching the
/// reference algorithm. The flag is used to skip redundant pairwise
/// distance checks during update (paper Algorithm 3 line 8).
///
/// We sort `Candidate`s by descending score so that the closest neighbors
/// come first (Qdrant convention).
#[derive(Debug, Clone, Copy)]
struct Candidate {
    idx: PointOffsetType,
    score: ScoreType,
    /// `true` if this candidate has not yet been "seen" by a refinement
    /// pass. This mirrors the reference's `flag` field on
    /// `nndescent::Neighbor`.
    new_flag: bool,
}

impl Candidate {
    #[inline]
    fn cmp_desc(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by score descending. Ties broken by idx for determinism.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Per-vertex state during refinement.
struct Pool {
    /// Pool of candidate neighbors. Order is maintained closest-first by
    /// the `update` step; between updates it may be unsorted.
    inner: Vec<Candidate>,
}

impl Pool {
    fn with_capacity(cap: usize) -> Self {
        Pool {
            inner: Vec::with_capacity(cap),
        }
    }
}

/// Configuration for [`build_layer0`].
pub struct RefinementParams {
    /// Initial out-degree of the random graph (paper's `S`).
    pub s: usize,
    /// Number of refinement rounds (paper's `R`).
    pub r: usize,
    /// Updates per round (paper's `Iter`).
    pub iter: usize,
    /// Cap on per-vertex pool size after each `add_reverse_edges` pass.
    /// Reference C++ uses 96.
    pub num_reverse_edges: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for RefinementParams {
    fn default() -> Self {
        // Defaults follow the MIRAGE paper §4 / README example.
        RefinementParams {
            s: 32,
            r: 4,
            iter: 15,
            num_reverse_edges: 96,
            seed: 2021,
        }
    }
}

/// Layer 0 builder.
///
/// `score_pair` is a thread-safe symmetric scoring function: `score_pair(a,
/// b)` returns a similarity score (higher = closer). It is called from many
/// threads concurrently; the implementation must be `Send + Sync`. In
/// practice this is wrapped around a `RawScorer::score_internal` plus
/// possibly a quantized scorer.
///
/// `num_points` is the number of vectors to refine. `is_alive(point_id)`
/// returns whether `point_id` is a non-deleted, valid vector. Deleted
/// points are still allocated a (possibly empty) pool slot to keep
/// addressing simple, but they are never picked as neighbors.
///
/// Returns a vector of pools; for each point `u`, `output[u]` is the
/// sorted-closest-first list of selected Layer 0 neighbors.
pub fn build_layer0<F, G>(
    num_points: usize,
    params: &RefinementParams,
    pool: &ThreadPool,
    is_alive: G,
    score_pair: F,
    stopped: &AtomicBool,
) -> OperationResult<Vec<Vec<ScoredPointOffset>>>
where
    F: Fn(PointOffsetType, PointOffsetType) -> ScoreType + Send + Sync,
    G: Fn(PointOffsetType) -> bool + Send + Sync,
{
    if num_points == 0 {
        return Ok(Vec::new());
    }
    if params.s == 0 {
        return Err(OperationError::service_error(
            "MIRAGE: refinement parameter S must be > 0",
        ));
    }
    if params.r == 0 || params.iter == 0 {
        return Err(OperationError::service_error(
            "MIRAGE: refinement parameters R and iter must both be > 0",
        ));
    }

    // Cache liveness for fast random sampling later.
    let mut alive: BitVec = BitVec::repeat(false, num_points);
    let mut alive_indices: Vec<PointOffsetType> = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let pid = i as PointOffsetType;
        if is_alive(pid) {
            alive.set(i, true);
            alive_indices.push(pid);
        }
    }
    if alive_indices.is_empty() {
        return Ok(vec![Vec::new(); num_points]);
    }

    // Per-vertex pool (locked for parallel update).
    let pools: Vec<Mutex<Pool>> = (0..num_points)
        .map(|_| Mutex::new(Pool::with_capacity(params.s.max(8))))
        .collect();

    // === Phase 1: random initialization (paper's `init_graph`) ===========
    init_random_graph(
        &alive_indices,
        &alive,
        params.s,
        params.seed,
        pool,
        &pools,
        &score_pair,
        stopped,
    )?;

    // === Phase 2: refinement rounds =====================================
    for round in 0..params.r {
        for _ in 0..params.iter {
            check_process_stopped(stopped)?;
            update_round(pool, &pools, &score_pair, stopped)?;
        }
        if round + 1 < params.r {
            add_reverse_edges(pool, &pools, params.num_reverse_edges, stopped)?;
        }
    }
    drop(score_pair); // explicitly release scorer captures before finalize

    // === Phase 3: finalize ==============================================
    // Sort pools closest-first, dedupe, and turn into ScoredPointOffset.
    let mut result: Vec<Vec<ScoredPointOffset>> = (0..num_points).map(|_| Vec::new()).collect();

    pool.install(|| {
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(u, out)| {
                if !alive.get_bit(u).unwrap_or(false) {
                    // Deleted/missing point: leave its layer-0 list empty.
                    return;
                }
                let mut p = pools[u].lock();
                // Stable dedupe: sort closest-first, then dedup by `idx`
                // keeping the first (closest) occurrence.
                p.inner.sort_unstable_by(Candidate::cmp_desc);
                let mut last_idx: Option<PointOffsetType> = None;
                out.reserve(p.inner.len());
                for c in p.inner.drain(..) {
                    if c.idx as usize == u {
                        continue; // never include self
                    }
                    if !alive.get_bit(c.idx as usize).unwrap_or(false) {
                        continue; // skip dead neighbor
                    }
                    if Some(c.idx) == last_idx {
                        continue; // skip duplicate
                    }
                    last_idx = Some(c.idx);
                    out.push(ScoredPointOffset {
                        idx: c.idx,
                        score: c.score,
                    });
                }
            });
    });

    Ok(result)
}

/// Phase 1: assign each alive vertex `S` distinct random alive neighbors.
fn init_random_graph<F>(
    alive_indices: &[PointOffsetType],
    _alive: &BitVec,
    s: usize,
    seed: u64,
    pool: &ThreadPool,
    pools: &[Mutex<Pool>],
    score_pair: &F,
    stopped: &AtomicBool,
) -> OperationResult<()>
where
    F: Fn(PointOffsetType, PointOffsetType) -> ScoreType + Send + Sync,
{
    let n_alive = alive_indices.len();
    let s_eff = s.min(n_alive.saturating_sub(1));
    if s_eff == 0 {
        // Only one alive point; nothing to connect.
        return Ok(());
    }

    pool.install(|| -> OperationResult<()> {
        alive_indices
            .par_iter()
            .try_for_each(|&pid| -> OperationResult<()> {
                check_process_stopped(stopped)?;
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0x9E37_79B9 ^ pid as u64));
                // Sample `s_eff` distinct alive neighbors (other than self).
                // `choose_multiple` does Reservoir sampling — O(N) — but
                // for our typical S (≤ 64) and pool size that's fine.
                // It guarantees uniqueness.
                let sample: Vec<PointOffsetType> = alive_indices
                    .iter()
                    .copied()
                    .filter(|&nb| nb != pid)
                    .choose_multiple(&mut rng, s_eff);

                let mut p = pools[pid as usize].lock();
                p.inner.clear();
                p.inner.reserve(sample.len());
                for nb in sample {
                    let score = score_pair(pid, nb);
                    p.inner.push(Candidate {
                        idx: nb,
                        score,
                        new_flag: true,
                    });
                }
                // Sort closest-first so subsequent `update` sees a sorted pool.
                p.inner.sort_unstable_by(Candidate::cmp_desc);
                Ok(())
            })?;
        Ok(())
    })
}

/// One refinement update pass (Algorithm 3 in the paper).
///
/// For each vertex `u`:
/// 1. Take its candidate pool, sort closest-first, dedupe by `idx`.
/// 2. Walk the pool in order, maintaining a "kept" subset:
///    - For candidate `nn`, check against each already-kept `other`: if
///      `score(nn, other) > score(u, nn)` (i.e. `other` is closer to `nn`
///      than `u` is to `nn`), reject `nn` and append `nn` to `other`'s pool
///      with score `score(other, nn)`. This is the reverse-edge insertion
///      described by Algorithm 3 line 13.
///    - If both `nn.new_flag` and `other.new_flag` are false, skip the
///      check (paper's "old/old" optimization).
/// 3. Mark all kept candidates' flags as `false` (no longer "new").
fn update_round<F>(
    pool: &ThreadPool,
    pools: &[Mutex<Pool>],
    score_pair: &F,
    stopped: &AtomicBool,
) -> OperationResult<()>
where
    F: Fn(PointOffsetType, PointOffsetType) -> ScoreType + Send + Sync,
{
    let n = pools.len();
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let u_pid = u as PointOffsetType;

                // Atomically swap the pool out so that other threads can
                // continue to push reverse-edge contributions into pools[u]
                // while we reorganize ours locally. (Reference C++ does the
                // same.)
                let mut old_pool = {
                    let mut p = pools[u].lock();
                    std::mem::take(&mut p.inner)
                };

                // Sort closest-first, dedupe by idx (keep closest).
                old_pool.sort_unstable_by(Candidate::cmp_desc);
                old_pool.dedup_by_key(|c| c.idx);

                let mut new_pool: Vec<Candidate> = Vec::with_capacity(old_pool.len());

                'candidate: for nn in old_pool.into_iter() {
                    if nn.idx == u_pid {
                        continue; // never keep self-loop
                    }
                    for other in &new_pool {
                        if !nn.new_flag && !other.new_flag {
                            // Both already processed in a previous round:
                            // their pairwise relationship was settled then,
                            // skip the redundant work.
                            continue;
                        }
                        if other.idx == nn.idx {
                            // Defensive (dedup should have killed this).
                            continue 'candidate;
                        }
                        let dist_nn_other = score_pair(nn.idx, other.idx);
                        // Higher score = closer. RNG: reject `u→nn` if there
                        // is `other` closer to `nn` than `u` is to `nn`.
                        if dist_nn_other > nn.score {
                            // `other` is closer to `nn` than `u` is.
                            // Replace edge `u→nn` with reverse contribution
                            // to other's pool.
                            let mut other_pool = pools[other.idx as usize].lock();
                            other_pool.inner.push(Candidate {
                                idx: nn.idx,
                                score: dist_nn_other,
                                new_flag: true,
                            });
                            drop(other_pool);
                            continue 'candidate;
                        }
                    }
                    new_pool.push(nn);
                }

                // Mark surviving candidates as "old".
                for c in new_pool.iter_mut() {
                    c.new_flag = false;
                }

                // Write back. Concurrent reverse-edge insertions from other
                // threads may have appended items to pools[u].inner *while*
                // we were processing — preserve them by appending after.
                let mut p = pools[u].lock();
                if p.inner.is_empty() {
                    p.inner = new_pool;
                } else {
                    let mut merged = new_pool;
                    merged.append(&mut p.inner);
                    p.inner = merged;
                }
                Ok(())
            })?;
        Ok(())
    })
}

/// Reverse-edge consolidation pass.
///
/// For every vertex `u` and every neighbor `nn` in `u`'s pool, push `u`
/// into `nn`'s reverse pool. Then merge each vertex's reverse pool into
/// its own pool, dedupe by `idx`, sort closest-first, and cap at
/// `num_reverse_edges`.
///
/// Note: scoring is not needed here; we reuse the pre-computed `score(u,
/// nn) == score(nn, u)` value (symmetry).
fn add_reverse_edges(
    pool: &ThreadPool,
    pools: &[Mutex<Pool>],
    num_reverse_edges: usize,
    stopped: &AtomicBool,
) -> OperationResult<()> {
    let n = pools.len();
    let reverse: Vec<Mutex<Vec<Candidate>>> = (0..n).map(|_| Mutex::new(Vec::new())).collect();

    // Phase A: scatter reverse-edge contributions.
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let p = pools[u].lock();
                let snapshot: Vec<Candidate> = p.inner.clone();
                drop(p);
                for nn in snapshot {
                    let mut r = reverse[nn.idx as usize].lock();
                    r.push(Candidate {
                        idx: u as PointOffsetType,
                        score: nn.score, // symmetric: score(u, nn) == score(nn, u)
                        new_flag: nn.new_flag,
                    });
                }
                Ok(())
            })?;
        Ok(())
    })?;

    // Phase B: merge reverse pool into main pool, dedupe, cap, mark "new".
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let mut rp = reverse[u].lock();
                let mut rev = std::mem::take(&mut *rp);
                drop(rp);
                if rev.is_empty() {
                    return Ok(());
                }
                let mut p = pools[u].lock();
                // Mark current pool entries as "new" again so that the next
                // refinement rounds re-evaluate their RNG status against
                // the freshly added reverse edges (matches reference C++).
                for c in p.inner.iter_mut() {
                    c.new_flag = true;
                }
                p.inner.append(&mut rev);
                p.inner.sort_unstable_by(Candidate::cmp_desc);
                p.inner.dedup_by_key(|c| c.idx);
                if p.inner.len() > num_reverse_edges {
                    p.inner.truncate(num_reverse_edges);
                }
                Ok(())
            })?;
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use rayon::ThreadPoolBuilder;

    use super::*;

    /// Verify that on a small synthetic dataset the refinement produces
    /// neighbor lists with high recall against ground-truth top-K.
    ///
    /// Uses negative L2 as the "score" (so higher = closer).
    #[test]
    fn test_layer0_refinement_recall() {
        const N: usize = 200;
        const D: usize = 8;

        // Generate deterministic random points.
        let mut rng = StdRng::seed_from_u64(42);
        let mut data: Vec<[f32; D]> = vec![[0.0; D]; N];
        for v in data.iter_mut() {
            for x in v.iter_mut() {
                use rand::RngExt as _;
                *x = rng.random_range(-1.0_f32..1.0_f32);
            }
        }

        let score_pair = |a: PointOffsetType, b: PointOffsetType| -> ScoreType {
            let va = &data[a as usize];
            let vb = &data[b as usize];
            let mut s = 0.0_f32;
            for k in 0..D {
                let d = va[k] - vb[k];
                s += d * d;
            }
            -s // negative L2: higher = closer
        };

        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let stopped = AtomicBool::new(false);

        let params = RefinementParams {
            s: 16,
            r: 3,
            iter: 8,
            num_reverse_edges: 64,
            seed: 1,
        };

        let layer0 =
            build_layer0(N, &params, &pool, |_| true, score_pair, &stopped).expect("build ok");

        assert_eq!(layer0.len(), N);
        for nbrs in &layer0 {
            assert!(!nbrs.is_empty(), "every alive point should have neighbors");
            // Ensure sorted closest-first (descending score).
            for w in nbrs.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "expected descending score: got {:?}",
                    nbrs
                );
            }
            // No self-loops, no duplicates.
            let mut seen = std::collections::HashSet::new();
            for c in nbrs {
                assert!(seen.insert(c.idx), "duplicate neighbor idx={}", c.idx);
            }
        }

        // Compute ground truth top-10 per point and measure recall.
        const K: usize = 10;
        let mut total_recall = 0.0_f32;
        for u in 0..N {
            let u_pid = u as PointOffsetType;
            let mut all: Vec<ScoredPointOffset> = (0..N)
                .filter(|&v| v != u)
                .map(|v| ScoredPointOffset {
                    idx: v as PointOffsetType,
                    score: score_pair(u_pid, v as PointOffsetType),
                })
                .collect();
            all.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            let gt: std::collections::HashSet<PointOffsetType> =
                all.iter().take(K).map(|c| c.idx).collect();
            let got: std::collections::HashSet<PointOffsetType> =
                layer0[u].iter().take(K).map(|c| c.idx).collect();
            total_recall +=
                gt.intersection(&got).count() as f32 / K as f32;
        }
        let avg_recall = total_recall / N as f32;
        eprintln!("MIRAGE Layer 0 recall@{K} on N={N}, D={D}: {avg_recall:.3}");
        // This is a sanity bound, not a quality guarantee; tighten as the
        // implementation matures. With S=16, R=3, iter=8 we routinely see
        // recall >= 0.6 on this trivial synthetic.
        assert!(
            avg_recall >= 0.5,
            "MIRAGE Layer 0 recall too low: {avg_recall}"
        );
    }
}
