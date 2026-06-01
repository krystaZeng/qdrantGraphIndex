//! Small FAISS-compatible random helpers used by the MIRAGE build path.
//!
//! FAISS uses `std::mt19937` directly in `MIRAGE.cpp` and wraps the same
//! generator in `RandomGenerator` for HNSW level assignment / shuffling. Rust's
//! `StdRng` is not MT19937, so we keep this tiny local implementation to make
//! the MIRAGE construction deterministic against the reference algorithm.

/// Minimal MT19937 implementation compatible with `std::mt19937(seed)`.
#[derive(Clone)]
pub(crate) struct FaissMt19937 {
    state: [u32; 624],
    index: usize,
}

impl FaissMt19937 {
    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..state.len() {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }

        Self { state, index: 624 }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }

        let mut y = self.state[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    pub(crate) fn rand_int(&mut self, max: usize) -> usize {
        debug_assert!(max > 0);
        self.next_u32() as usize % max
    }

    pub(crate) fn rand_float(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn twist(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7FFF_FFFF;
        const MATRIX_A: u32 = 0x9908_B0DF;

        for i in 0..N {
            let x = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MATRIX_A;
            }
            self.state[i] = self.state[(i + M) % N] ^ xa;
        }
        self.index = 0;
    }
}

pub(crate) fn faiss_random_level(m: usize, rng: &mut FaissMt19937) -> usize {
    let m = m.max(2);
    let level_mult = 1.0_f32 / (m as f32).ln();
    let mut f = rng.rand_float();

    for level in 0usize.. {
        let proba = (-(level as f32) / level_mult).exp() * (1.0 - (-1.0 / level_mult).exp());
        if proba < 1e-9 {
            return level.saturating_sub(1);
        }
        if f < proba {
            return level;
        }
        f -= proba;
    }

    unreachable!()
}

pub(crate) fn faiss_shuffle<T>(values: &mut [T], rng: &mut FaissMt19937) {
    for j in 0..values.len() {
        let offset = rng.rand_int(values.len() - j);
        values.swap(j, j + offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mt19937_matches_reference_seed() {
        let mut rng = FaissMt19937::new(5489);
        let expected = [
            3_499_211_612,
            581_869_302,
            3_890_346_734,
            3_586_334_585,
            545_404_204,
        ];

        for value in expected {
            assert_eq!(rng.next_u32(), value);
        }
    }

    #[test]
    fn test_faiss_shuffle_is_stable() {
        let mut values = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut rng = FaissMt19937::new(789);
        faiss_shuffle(&mut values, &mut rng);

        assert_eq!(values, [3, 4, 1, 2, 5, 7, 6, 0]);
    }
}
