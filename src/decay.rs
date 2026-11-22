//! Time-decay scoring (PRD §4, Layer 3).
//!
//! `score = similarity × factor`, where
//!
//! * `freshness = 0.5^(age / half_life)` ∈ (0, 1]
//! * `time_mult = floor + (1 − floor)·freshness + boost·freshness`
//!   — fresh memories get up to `1 + boost`, very old ones tend to `floor`;
//! * `protect = smoothstep(protect_lo, protect_hi, similarity)` — near-perfect
//!   matches are shielded from the *penalty* (never from the boost):
//!   `factor = lerp(time_mult, max(time_mult, 1), protect)`.
//!
//! All functions are pure.

pub const DAY_MS: u64 = 86_400_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayParams {
    pub enabled: bool,
    pub half_life_ms: u64,
    /// Multiplier approached by infinitely old memories.
    pub floor: f32,
    /// Extra multiplier for brand-new memories.
    pub boost: f32,
    /// Similarity at which penalty protection starts…
    pub protect_lo: f32,
    /// …and at which it is complete.
    pub protect_hi: f32,
}

impl Default for DecayParams {
    fn default() -> Self {
        DecayParams {
            enabled: true,
            half_life_ms: 30 * DAY_MS,
            floor: 0.5,
            boost: 0.1,
            protect_lo: 0.92,
            protect_hi: 0.99,
        }
    }
}

/// `0.5^(age/half_life)`; 1.0 for non-positive ages or a zero half-life.
pub fn freshness(age_ms: u64, half_life_ms: u64) -> f32 {
    if half_life_ms == 0 {
        return 1.0;
    }
    (0.5f64).powf(age_ms as f64 / half_life_ms as f64) as f32
}

/// Hermite smoothstep of `x` between `lo` and `hi` (step function if equal).
pub fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    if hi <= lo {
        return if x >= hi { 1.0 } else { 0.0 };
    }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Multiplicative time factor for a memory of `age_ms` with `similarity`.
pub fn factor(similarity: f32, age_ms: u64, p: &DecayParams) -> f32 {
    if !p.enabled {
        return 1.0;
    }
    let f = freshness(age_ms, p.half_life_ms);
    let floor = p.floor.clamp(0.0, 1.0);
    let time_mult = floor + (1.0 - floor) * f + p.boost.max(0.0) * f;
    let protect = smoothstep(p.protect_lo, p.protect_hi, similarity);
    time_mult + (time_mult.max(1.0) - time_mult) * protect
}

/// Final ranking score. Negative similarities are left untouched so that
/// decay can never lift an anti-correlated memory.
pub fn score(similarity: f32, age_ms: u64, p: &DecayParams) -> f32 {
    if similarity <= 0.0 {
        return similarity;
    }
    similarity * factor(similarity, age_ms, p)
}

/// Age of a memory updated at `updated_at` as seen at `now` (both unix ms).
pub fn age(now_ms: u64, updated_at: u64) -> u64 {
    now_ms.saturating_sub(updated_at)
}

/// Does a memory carrying `tags` decay? With no `patterns` every memory
/// does (the default); otherwise only those with a matching tag, so durable
/// knowledge is ranked by similarity alone. A pattern ending in `*` matches
/// by prefix; case is ignored.
pub fn decays<'a>(patterns: &[String], tags: impl IntoIterator<Item = &'a str>) -> bool {
    if patterns.is_empty() {
        return true;
    }
    tags.into_iter().any(|t| {
        let t = t.to_ascii_lowercase();
        patterns.iter().any(|p| {
            let p = p.trim().to_ascii_lowercase();
            match p.strip_suffix('*') {
                Some(pre) => t.starts_with(pre),
                None => t == p,
            }
        })
    })
}

/// `p` for a memory that decays, decay switched off otherwise.
pub fn params_for(p: &DecayParams, decays: bool) -> DecayParams {
    if decays { *p } else { DecayParams { enabled: false, ..*p } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_halves_every_half_life() {
        assert_eq!(freshness(0, DAY_MS), 1.0);
        assert!((freshness(DAY_MS, DAY_MS) - 0.5).abs() < 1e-6);
        assert!((freshness(3 * DAY_MS, DAY_MS) - 0.125).abs() < 1e-6);
        assert_eq!(freshness(5, 0), 1.0);
    }

    #[test]
    fn fresh_beats_old_at_equal_similarity() {
        let p = DecayParams::default();
        let fresh = score(0.8, DAY_MS, &p);
        let old = score(0.8, 180 * DAY_MS, &p);
        assert!(fresh > 0.8 && old < 0.8 * 0.6, "{fresh} {old}");
    }

    #[test]
    fn near_perfect_old_match_is_not_penalized() {
        let p = DecayParams::default();
        let old_exact = score(0.999, 365 * DAY_MS, &p);
        assert!((old_exact - 0.999).abs() < 1e-3);
        // …and beats a fresh but merely related memory.
        assert!(old_exact > score(0.8, 0, &p));
    }

    #[test]
    fn relevance_still_dominates_extreme_age_gaps() {
        let p = DecayParams::default();
        // An old but clearly relevant memory beats a fresh unrelated one…
        assert!(score(0.85, 365 * DAY_MS, &p) > score(0.35, 0, &p));
        // …and a fresh near-match only beats an old exact match inside the
        // protection band.
        assert!(score(1.0, 365 * DAY_MS, &p) > score(0.9, 0, &p));
    }

    #[test]
    fn factor_is_monotone_in_age_and_bounded() {
        let p = DecayParams::default();
        let mut prev = f32::INFINITY;
        for days in [0u64, 1, 7, 30, 90, 365, 3650] {
            let f = factor(0.5, days * DAY_MS, &p);
            assert!(f <= prev && f >= p.floor - 1e-6 && f <= 1.0 + p.boost + 1e-6);
            prev = f;
        }
        let off = DecayParams { enabled: false, ..p };
        assert_eq!(factor(0.5, 1000 * DAY_MS, &off), 1.0);
        assert_eq!(score(-0.3, 1000 * DAY_MS, &p), -0.3);
    }

    #[test]
    fn decay_tags_select_volatile_memories() {
        let pats = vec!["status".to_string(), "ibkr-week*".to_string()];
        assert!(decays(&[], ["anything"]));
        assert!(decays(&[], []));
        assert!(decays(&pats, ["family", "Status"]));
        assert!(decays(&pats, ["ibkr-week-47"]));
        assert!(!decays(&pats, ["family", "statuses"]));
        assert!(!decays(&pats, []));
        let p = DecayParams::default();
        assert_eq!(factor(0.5, 400 * DAY_MS, &params_for(&p, false)), 1.0);
        assert!(factor(0.5, 400 * DAY_MS, &params_for(&p, true)) < 1.0);
    }

    #[test]
    fn smoothstep_edges() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.5), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
    }
}
