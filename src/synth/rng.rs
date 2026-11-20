//! Deterministic PRNG (xoshiro256**) for synthetic data, fuzzing and tests.
//! Explicitly seeded: there is no global RNG state.

use crate::hash::mix64;

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            *slot = mix64(x);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn f32(&mut self) -> f32 {
        self.f64() as f32
    }

    /// Uniform integer in `[0, n)`; returns 0 when `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f64().max(f64::MIN_POSITIVE);
        let u2 = self.f64();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// Random unit vector of dimension `dim`.
    pub fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.normal()).collect();
        crate::math::normalize(&v).unwrap_or_else(|_| {
            let mut e = vec![0.0; dim.max(1)];
            e[0] = 1.0;
            e
        })
    }

    /// `base + noise * gaussian`, renormalized.
    pub fn perturb(&mut self, base: &[f32], noise: f32) -> Vec<f32> {
        let v: Vec<f32> = base.iter().map(|x| x + noise * self.normal()).collect();
        crate::math::normalize(&v).unwrap_or_else(|_| base.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let (mut a, mut b) = (Rng::new(9), Rng::new(9));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        for _ in 0..1000 {
            let x = a.f64();
            assert!((0.0..1.0).contains(&x));
            assert!(a.below(7) < 7);
        }
        assert_eq!(a.below(0), 0);
        let mean: f32 = (0..10_000).map(|_| a.normal()).sum::<f32>() / 10_000.0;
        assert!(mean.abs() < 0.05);
    }
}
