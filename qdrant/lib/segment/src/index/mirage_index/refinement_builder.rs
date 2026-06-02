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
//! 1. init_random_graph: each vertex u gets S random neighbors using the same
//!    unique integer generation pattern as the local C++ MIRAGE implementation,
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
use rayon::ThreadPool;
use rayon::prelude::*;

use super::faiss_random::FaissMt19937;
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};

pub trait PairScorer {
    fn score_pair(&mut self, a: PointOffsetType, b: PointOffsetType) -> ScoreType;
}

impl<F> PairScorer for F
where
    F: FnMut(PointOffsetType, PointOffsetType) -> ScoreType,
{
    fn score_pair(&mut self, a: PointOffsetType, b: PointOffsetType) -> ScoreType {
        self(a, b)
    }
}

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
        // Defaults match the local C++ MIRAGE implementation.
        RefinementParams {
            s: 16,
            r: 4,
            iter: 15,
            num_reverse_edges: 96,
            seed: 2021,
        }
    }
}

/// Whether this run can be compared with the C++ MIRAGE random initialization.
///
/// The reference C++ implementation assumes `ntotal > S`; for `N <= S`, Qdrant
/// uses a complete non-self graph as a safe boundary behavior instead.
pub(crate) fn cpp_parity_eligible(n_alive: usize, s: usize) -> bool {
    n_alive > s
}

/// Minimum guard for reporting formal recall acceptance from MIRAGE tests.
///
/// `N > S` is required for C++ parity. `N >= 10*S` keeps tiny-but-parity-valid
/// datasets (for example `N = S + 1`) out of formal recall conclusions.
pub(crate) fn recall_acceptance_eligible(n_alive: usize, s: usize) -> bool {
    s > 0 && cpp_parity_eligible(n_alive, s) && n_alive >= s.saturating_mul(10)
}

/// Layer 0 builder.
///
/// `make_scorer` creates a thread-local symmetric scorer. Each worker/chunk
/// gets its own scorer and reuses it for many pairwise comparisons, matching
/// the reference implementation's thread-local distance-computer model.
///
/// `num_points` is the number of vectors to refine. `is_alive(point_id)`
/// returns whether `point_id` is a non-deleted, valid vector. Deleted
/// points are still allocated a (possibly empty) pool slot to keep
/// addressing simple, but they are never picked as neighbors.
///
/// Returns a vector of pools; for each point `u`, `output[u]` is the
/// sorted-closest-first list of selected Layer 0 neighbors.
pub fn build_layer0<MakeScorer, Scorer, G>(
    num_points: usize,
    params: &RefinementParams,
    pool: &ThreadPool,
    is_alive: G,
    make_scorer: MakeScorer,
    stopped: &AtomicBool,
) -> OperationResult<Vec<Vec<ScoredPointOffset>>>
where
    MakeScorer: Fn() -> OperationResult<Scorer> + Sync,
    Scorer: PairScorer,
    G: Fn(PointOffsetType) -> bool + Send + Sync,
{
    if num_points == 0 {
        return Ok(Vec::new());
    }
    if num_points > PointOffsetType::MAX as usize {
        return Err(OperationError::service_error(
            "MIRAGE: too many vectors for PointOffsetType",
        ));
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
    //
    // Internally MIRAGE's C++ source indexes a dense `0..ntotal` graph. Qdrant
    // internal point ids can have holes due to deletes, so we run refinement on
    // dense logical ids and map them back to real PointOffsetType values during
    // scoring/finalization.
    let pools: Vec<Mutex<Pool>> = (0..alive_indices.len())
        .map(|_| Mutex::new(Pool::with_capacity(params.s.max(8))))
        .collect();

    // === Phase 1: random initialization (paper's `init_graph`) ===========
    init_random_graph(
        &alive_indices,
        params.s,
        params.seed,
        pool,
        &pools,
        &make_scorer,
        stopped,
    )?;

    // === Phase 2: refinement rounds =====================================
    for round in 0..params.r {
        for _ in 0..params.iter {
            check_process_stopped(stopped)?;
            update_round(pool, &alive_indices, &pools, &make_scorer, stopped)?;
        }
        if round + 1 < params.r {
            add_reverse_edges(pool, &pools, params.num_reverse_edges, stopped)?;
        }
    }
    drop(make_scorer); // explicitly release scorer captures before finalize

    // === Phase 3: finalize ==============================================
    // Sort pools closest-first, dedupe, and turn into ScoredPointOffset.
    let mut alive_result: Vec<Vec<ScoredPointOffset>> =
        (0..alive_indices.len()).map(|_| Vec::new()).collect();

    pool.install(|| {
        alive_result
            .par_iter_mut()
            .enumerate()
            .for_each(|(logical_u, out)| {
                let mut p = pools[logical_u].lock();
                // Stable dedupe: sort closest-first, then dedup by `idx`
                // keeping the first (closest) occurrence.
                p.inner.sort_unstable_by(Candidate::cmp_desc);
                let mut last_idx: Option<PointOffsetType> = None;
                out.reserve(p.inner.len());
                for c in p.inner.drain(..) {
                    if c.idx as usize == logical_u {
                        continue; // never include self
                    }
                    let Some(&actual_idx) = alive_indices.get(c.idx as usize) else {
                        continue; // skip defensive out-of-range candidate
                    };
                    if !alive.get_bit(actual_idx as usize).unwrap_or(false) {
                        continue; // skip defensive dead neighbor
                    }
                    if Some(c.idx) == last_idx {
                        continue; // skip duplicate
                    }
                    last_idx = Some(c.idx);
                    out.push(ScoredPointOffset {
                        idx: actual_idx,
                        score: c.score,
                    });
                }
            });
    });

    let mut result: Vec<Vec<ScoredPointOffset>> = (0..num_points).map(|_| Vec::new()).collect();
    for (logical, actual) in alive_indices.iter().copied().enumerate() {
        result[actual as usize] = std::mem::take(&mut alive_result[logical]);
    }

    Ok(result)
}

fn gen_random_mirage_style(rng: &mut FaissMt19937, size: usize, n: usize) -> Vec<usize> {
    if size == 0 || n == 0 {
        return Vec::new();
    }

    debug_assert!(size < n);
    let upper = n - size;
    let mut addr: Vec<usize> = (0..size).map(|_| rng.rand_int(upper)).collect();
    addr.sort_unstable();

    for i in 1..addr.len() {
        if addr[i] <= addr[i - 1] {
            addr[i] = addr[i - 1] + 1;
        }
    }

    let offset = rng.rand_int(n);
    for item in &mut addr {
        *item = (*item + offset) % n;
    }

    addr
}

fn chunk_bounds(thread_id: usize, num_threads: usize, len: usize) -> (usize, usize) {
    let base = len / num_threads;
    let rem = len % num_threads;
    let start = thread_id * base + thread_id.min(rem);
    let end = start + base + usize::from(thread_id < rem);
    (start, end)
}

/// Phase 1: assign each alive vertex MIRAGE-style random alive neighbors.
fn init_random_graph<MakeScorer, Scorer>(
    alive_indices: &[PointOffsetType],
    s: usize,
    seed: u64,
    pool: &ThreadPool,
    pools: &[Mutex<Pool>],
    make_scorer: &MakeScorer,
    stopped: &AtomicBool,
) -> OperationResult<()>
where
    MakeScorer: Fn() -> OperationResult<Scorer> + Sync,
    Scorer: PairScorer,
{
    let n_alive = alive_indices.len();
    if n_alive <= 1 {
        // Only one alive point; nothing to connect.
        return Ok(());
    }

    let num_threads = pool.current_num_threads().max(1).min(n_alive);

    if !cpp_parity_eligible(n_alive, s) {
        // The C++ implementation assumes `ntotal > S` because `gen_random`
        // samples from `N - S`. Qdrant can build tiny segments, so for this
        // boundary we use the complete non-self graph as the safe equivalent.
        pool.install(|| -> OperationResult<()> {
            (0..num_threads)
                .into_par_iter()
                .try_for_each(|thread_id| -> OperationResult<()> {
                    let (start, end) = chunk_bounds(thread_id, num_threads, n_alive);
                    let mut scorer = make_scorer()?;

                    for logical_u in start..end {
                        if logical_u % 4096 == 0 {
                            check_process_stopped(stopped)?;
                        }
                        let actual_u = alive_indices[logical_u];
                        let mut p = pools[logical_u].lock();
                        p.inner.clear();
                        p.inner.reserve(n_alive.saturating_sub(1));
                        for (logical_nb, &actual_nb) in alive_indices.iter().enumerate() {
                            if logical_nb == logical_u {
                                continue;
                            }
                            p.inner.push(Candidate {
                                idx: logical_nb as PointOffsetType,
                                score: scorer.score_pair(actual_u, actual_nb),
                                new_flag: true,
                            });
                        }
                        p.inner.sort_unstable_by(Candidate::cmp_desc);
                    }
                    Ok(())
                })?;
            Ok(())
        })?;
        return Ok(());
    }

    pool.install(|| -> OperationResult<()> {
        (0..num_threads)
            .into_par_iter()
            .try_for_each(|thread_id| -> OperationResult<()> {
                let (start, end) = chunk_bounds(thread_id, num_threads, n_alive);
                let mut scorer = make_scorer()?;

                let mut rng = FaissMt19937::new(
                    seed.wrapping_mul(7741).wrapping_add(thread_id as u64) as u32,
                );

                for logical_u in start..end {
                    if logical_u % 4096 == 0 {
                        check_process_stopped(stopped)?;
                    }
                    let actual_u = alive_indices[logical_u];
                    let sample_positions = gen_random_mirage_style(&mut rng, s, n_alive);

                    let mut p = pools[logical_u].lock();
                    p.inner.clear();
                    p.inner.reserve(sample_positions.len());
                    for logical_nb in sample_positions {
                        if logical_nb == logical_u {
                            continue;
                        }
                        let actual_nb = alive_indices[logical_nb];
                        let score = scorer.score_pair(actual_u, actual_nb);
                        p.inner.push(Candidate {
                            idx: logical_nb as PointOffsetType,
                            score,
                            new_flag: true,
                        });
                    }
                    // Sort closest-first so subsequent `update` sees a sorted pool.
                    p.inner.sort_unstable_by(Candidate::cmp_desc);
                }
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
fn update_round<MakeScorer, Scorer>(
    pool: &ThreadPool,
    alive_indices: &[PointOffsetType],
    pools: &[Mutex<Pool>],
    make_scorer: &MakeScorer,
    stopped: &AtomicBool,
) -> OperationResult<()>
where
    MakeScorer: Fn() -> OperationResult<Scorer> + Sync,
    Scorer: PairScorer,
{
    let n = pools.len();
    let num_threads = pool.current_num_threads().max(1).min(n.max(1));
    pool.install(|| -> OperationResult<()> {
        (0..num_threads)
            .into_par_iter()
            .try_for_each(|thread_id| -> OperationResult<()> {
                let (start, end) = chunk_bounds(thread_id, num_threads, n);
                let mut scorer = make_scorer()?;

                for u in start..end {
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
                            let dist_nn_other = scorer.score_pair(
                                alive_indices[nn.idx as usize],
                                alive_indices[other.idx as usize],
                            );
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

                    // Match the local C++ MIRAGE implementation: assign the newly
                    // selected pool directly. Reverse insertions that raced into
                    // the pool before this assignment are overwritten there too.
                    let mut p = pools[u].lock();
                    p.inner = new_pool;
                }
                Ok(())
            })?;
        Ok(())
    })
}

/// Reverse-edge consolidation pass, matching local C++ `Mirage::add_reverse_edges`.
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

    // Phase A: scatter reverse-edge contributions into reverse[nn].
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
                    if nn.idx as usize >= n {
                        continue;
                    }
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

    // Phase B: append each current pool into its own reverse pool, mark those
    // entries new, then sort/dedup/truncate the reverse pool. The current graph
    // pools are intentionally emptied here, as in the C++ source.
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let mut current = {
                    let mut p = pools[u].lock();
                    std::mem::take(&mut p.inner)
                };
                for c in current.iter_mut() {
                    c.new_flag = true;
                }

                let mut rp = reverse[u].lock();
                rp.append(&mut current);
                rp.sort_unstable_by(Candidate::cmp_desc);
                rp.dedup_by_key(|c| c.idx);
                if rp.len() > num_reverse_edges {
                    rp.truncate(num_reverse_edges);
                }
                Ok(())
            })?;
        Ok(())
    })?;

    // Phase C: backfill reverse pools into graph[nn].
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let snapshot = {
                    let rp = reverse[u].lock();
                    rp.clone()
                };
                for nn in snapshot {
                    if nn.idx as usize >= n {
                        continue;
                    }
                    let mut p = pools[nn.idx as usize].lock();
                    p.inner.push(Candidate {
                        idx: u as PointOffsetType,
                        score: nn.score,
                        new_flag: nn.new_flag,
                    });
                }
                Ok(())
            })?;
        Ok(())
    })?;

    // Phase D: final sort/truncate. C++ does not deduplicate at this point;
    // final build output deduplicates after all rounds complete.
    pool.install(|| -> OperationResult<()> {
        (0..n)
            .into_par_iter()
            .try_for_each(|u| -> OperationResult<()> {
                if u % 4096 == 0 {
                    check_process_stopped(stopped)?;
                }
                let mut p = pools[u].lock();
                p.inner.sort_unstable_by(Candidate::cmp_desc);
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
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use crate::common::operation_error::OperationError;
    use crate::index::mirage_index::golden::{generate_vectors, load_cpp_golden_n64_d8_l2};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rayon::ThreadPoolBuilder;

    use super::*;

    struct NoopScorer;

    impl PairScorer for NoopScorer {
        fn score_pair(&mut self, _a: PointOffsetType, _b: PointOffsetType) -> ScoreType {
            0.0
        }
    }

    fn failing_scorer_factory() -> OperationResult<NoopScorer> {
        Err(OperationError::service_error_light(
            "MIRAGE test scorer factory failed",
        ))
    }

    fn assert_scorer_factory_error(err: OperationError) {
        assert!(matches!(
            err,
            OperationError::ServiceError { description, .. }
                if description == "MIRAGE test scorer factory failed"
        ));
    }

    #[test]
    fn test_gen_random_mirage_style_returns_unique_values() {
        let mut rng = FaissMt19937::new(42);
        let values = gen_random_mirage_style(&mut rng, 16, 200);

        assert_eq!(values.len(), 16);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), values.len());
        assert!(values.iter().all(|&value| value < 200));
    }

    #[test]
    fn test_cpp_parity_and_recall_acceptance_guards() {
        const S: usize = 16;

        assert!(!cpp_parity_eligible(S, S));
        assert!(cpp_parity_eligible(S + 1, S));

        assert!(!recall_acceptance_eligible(S, S));
        assert!(!recall_acceptance_eligible(S + 1, S));
        assert!(recall_acceptance_eligible(10 * S, S));
    }

    #[test]
    fn test_build_layer0_returns_error_when_scorer_factory_fails_cpp_parity_path() {
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let stopped = AtomicBool::new(false);
        let params = RefinementParams {
            s: 4,
            r: 1,
            iter: 1,
            num_reverse_edges: 8,
            seed: 1,
        };

        let err = build_layer0(
            32,
            &params,
            &pool,
            |_| true,
            failing_scorer_factory,
            &stopped,
        )
        .unwrap_err();

        assert_scorer_factory_error(err);
    }

    #[test]
    fn test_build_layer0_returns_error_when_scorer_factory_fails_small_n_path() {
        const N: usize = 8;
        const S: usize = 8;

        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let stopped = AtomicBool::new(false);
        let params = RefinementParams {
            s: S,
            r: 1,
            iter: 1,
            num_reverse_edges: 8,
            seed: 1,
        };

        assert!(!cpp_parity_eligible(N, S));
        let err = build_layer0(
            N,
            &params,
            &pool,
            |_| true,
            failing_scorer_factory,
            &stopped,
        )
        .unwrap_err();

        assert_scorer_factory_error(err);
    }

    #[test]
    fn test_qdrant_small_n_boundary_uses_complete_non_self_graph() {
        const N: usize = 8;
        const S: usize = 8;

        let alive_indices: Vec<PointOffsetType> = (0..N as PointOffsetType).collect();
        let pools: Vec<Mutex<Pool>> = (0..N).map(|_| Mutex::new(Pool::with_capacity(S))).collect();
        let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let stopped = AtomicBool::new(false);
        let score_pair = |a: PointOffsetType, b: PointOffsetType| -> ScoreType {
            -((a as ScoreType) - (b as ScoreType)).abs()
        };

        assert!(!cpp_parity_eligible(N, S));
        init_random_graph(
            &alive_indices,
            S,
            2021,
            &pool,
            &pools,
            &|| Ok(&score_pair),
            &stopped,
        )
        .expect("small-N boundary initialization should succeed");

        for (u, pool) in pools.iter().enumerate() {
            let pool = pool.lock();
            assert_eq!(
                pool.inner.len(),
                N - 1,
                "Qdrant small-N boundary behavior uses a complete non-self graph",
            );

            let mut seen = std::collections::HashSet::new();
            for candidate in &pool.inner {
                assert_ne!(
                    candidate.idx as usize, u,
                    "small-N boundary must not create self-loop"
                );
                assert!(
                    seen.insert(candidate.idx),
                    "small-N boundary must not create duplicate candidate {}",
                    candidate.idx,
                );
            }
            assert_eq!(seen.len(), N - 1);
        }
    }

    /// Verify that on a small synthetic dataset the refinement produces a
    /// structurally valid RNG-pruned Layer 0 graph. Product recall is tested at
    /// the segment search level, not by treating Layer 0 out-edges as exact KNN.
    #[test]
    fn test_layer0_refinement_structural_invariants() {
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

        let layer0 = build_layer0(N, &params, &pool, |_| true, || Ok(&score_pair), &stopped)
            .expect("build ok");

        assert_eq!(layer0.len(), N);
        for (u, nbrs) in layer0.iter().enumerate() {
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
                assert_ne!(c.idx as usize, u, "self-loop for point {u}");
                assert!(seen.insert(c.idx), "duplicate neighbor idx={}", c.idx);
            }
        }

        let non_empty = layer0.iter().filter(|nbrs| !nbrs.is_empty()).count();
        assert!(
            non_empty > N * 9 / 10,
            "expected most alive points to have neighbors, got {non_empty}/{N}",
        );
    }

    #[test]
    fn test_layer0_refinement_matches_cpp_golden() {
        struct FixtureScorer<'a> {
            vectors: &'a [Vec<f32>],
        }

        impl PairScorer for FixtureScorer<'_> {
            fn score_pair(&mut self, a: PointOffsetType, b: PointOffsetType) -> ScoreType {
                -self.vectors[a as usize]
                    .iter()
                    .zip(&self.vectors[b as usize])
                    .map(|(left, right)| {
                        let diff = left - right;
                        diff * diff
                    })
                    .sum::<ScoreType>()
            }
        }

        let fixture = load_cpp_golden_n64_d8_l2();
        let vectors = generate_vectors(&fixture);
        let pool = ThreadPoolBuilder::new()
            .num_threads(fixture.params.threads)
            .build()
            .unwrap();
        let stopped = AtomicBool::new(false);
        let params = RefinementParams {
            s: fixture.params.s,
            r: fixture.params.r,
            iter: fixture.params.iter,
            num_reverse_edges: fixture.params.num_reverse_edges,
            seed: fixture.params.mirage_seed,
        };

        let layer0 = build_layer0(
            fixture.n,
            &params,
            &pool,
            |_| true,
            || Ok(FixtureScorer { vectors: &vectors }),
            &stopped,
        )
        .expect("MIRAGE layer0 golden build should succeed");

        let actual: Vec<Vec<usize>> = layer0
            .iter()
            .map(|neighbors| {
                neighbors
                    .iter()
                    .map(|neighbor| neighbor.idx as usize)
                    .collect()
            })
            .collect();

        // The authoritative C++ final_graph order can differ for candidates
        // with equivalent refinement priority. The semantic contract for
        // Layer 0 parity is the out-neighbor set; the final injected HNSW
        // neighbor order is checked separately against the C++ hierarchy dump.
        let actual_sets: Vec<BTreeSet<usize>> = actual
            .iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let expected_sets: Vec<BTreeSet<usize>> = fixture
            .layer0
            .iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();

        assert_eq!(actual_sets, expected_sets);
    }

    #[test]
    fn test_layer0_refinement_reuses_scorer_per_worker_chunk() {
        const N: usize = 96;
        const D: usize = 8;

        struct CountingScorer {
            data: Arc<Vec<[f32; D]>>,
            score_count: Arc<AtomicUsize>,
        }

        impl PairScorer for CountingScorer {
            fn score_pair(&mut self, a: PointOffsetType, b: PointOffsetType) -> ScoreType {
                self.score_count.fetch_add(1, Ordering::Relaxed);
                let va = &self.data[a as usize];
                let vb = &self.data[b as usize];
                let mut s = 0.0_f32;
                for k in 0..D {
                    let d = va[k] - vb[k];
                    s += d * d;
                }
                -s
            }
        }

        let data = Arc::new(
            (0..N)
                .map(|i| {
                    let mut v = [0.0; D];
                    for (k, value) in v.iter_mut().enumerate() {
                        *value = (i * 17 + k * 31) as f32 / 100.0;
                    }
                    v
                })
                .collect::<Vec<_>>(),
        );
        let scorer_init_count = Arc::new(AtomicUsize::new(0));
        let score_count = Arc::new(AtomicUsize::new(0));

        let make_scorer = || {
            scorer_init_count.fetch_add(1, Ordering::Relaxed);
            Ok(CountingScorer {
                data: Arc::clone(&data),
                score_count: Arc::clone(&score_count),
            })
        };

        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let stopped = AtomicBool::new(false);
        let params = RefinementParams {
            s: 12,
            r: 2,
            iter: 4,
            num_reverse_edges: 48,
            seed: 7,
        };

        build_layer0(N, &params, &pool, |_| true, make_scorer, &stopped).expect("build ok");

        let scorer_inits = scorer_init_count.load(Ordering::Relaxed);
        let scores = score_count.load(Ordering::Relaxed);
        assert!(scores > N * params.s);
        assert!(
            scorer_inits * 20 < scores,
            "expected scorer reuse per worker chunk, got {scorer_inits} scorer initializations for {scores} scores",
        );
    }
}
