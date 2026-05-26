use crate::coeff::Coeff;
use crate::fixes::Fixes;
use crate::solver::bnb::PartiallyAssignedTerm;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalCouplingKind {
    /// `s_j = alpha * s_i` can be inferred when `s_i` is assigned.
    JGivenI,
    /// `s_i = alpha * s_j` can be inferred when `s_j` is assigned.
    IGivenJ,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalCoupling<C: Coeff> {
    pub i: usize,
    pub j: usize,
    pub alpha: C,
    pub kind: ConditionalCouplingKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PairBounds<C: Coeff> {
    a_min: C,
    a_max: C,
    has_a: bool,
    has_b: bool,
    has_c: bool,
}

fn term_interval<C: Coeff>(coeff: C, degree_after_elimination: usize) -> (C, C) {
    if degree_after_elimination == 0 {
        (coeff, coeff)
    } else {
        let abs = coeff.abs();
        (-abs, abs)
    }
}

/// Compute the bounds for the coupling term `A` in the decomposition of `f` w.r.t. variables `i` and `j`.
/// This is used to detect variable pairs that are unconditionally or conditionally coupled.
fn pair_bounds<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    i: usize,
    j: usize,
) -> PairBounds<C> {
    let mut a_min = C::zero();
    let mut a_max = C::zero();
    let mut has_a = false;
    let mut has_b = false;
    let mut has_c = false;

    for term in &instance.terms {
        let has_i = term.indices.binary_search(&i).is_ok();
        let has_j = term.indices.binary_search(&j).is_ok();

        if !has_i && !has_j {
            continue;
        }

        let reduced_degree = term.indices.len() - usize::from(has_i) - usize::from(has_j);
        let (lo, hi) = term_interval(term.coeff, reduced_degree);

        match (has_i, has_j) {
            (true, true) => {
                has_a = true;
                a_min += lo;
                a_max += hi;
            }
            (true, false) => {
                has_b = true;
            }
            (false, true) => {
                has_c = true;
            }
            (false, false) => unreachable!(),
        }
    }

    PairBounds {
        a_min,
        a_max,
        has_a,
        has_b,
        has_c,
    }
}

fn pair_bounds_active<C: Coeff, V: VarDomain>(
    _instance: &HuboInstance<C, V>,
    term_status: &[Option<PartiallyAssignedTerm<C>>],
    i: usize,
    j: usize,
) -> PairBounds<C> {
    let mut a_min = C::zero();
    let mut a_max = C::zero();
    let mut has_a = false;
    let mut has_b = false;
    let mut has_c = false;

    for status in term_status.iter().flatten() {
        let has_i = status.free_variables.binary_search(&i).is_ok();
        let has_j = status.free_variables.binary_search(&j).is_ok();

        if !has_i && !has_j {
            continue;
        }

        let coeff = status.coeff;
        let reduced_degree = status.free_variables.len() - usize::from(has_i) - usize::from(has_j);
        let (lo, hi) = term_interval(coeff, reduced_degree);

        match (has_i, has_j) {
            (true, true) => {
                has_a = true;
                a_min += lo;
                a_max += hi;
            }
            (true, false) => {
                has_b = true;
            }
            (false, true) => {
                has_c = true;
            }
            (false, false) => unreachable!(),
        }
    }

    PairBounds {
        a_min,
        a_max,
        has_a,
        has_b,
        has_c,
    }
}

fn alpha_from_bounds<C: Coeff>(a_min: C, a_max: C) -> Option<C> {
    // A > 0: optimal s_i*s_j = -1 → s_j = -s_i → alpha = -1
    // A < 0: optimal s_i*s_j = +1 → s_j = +s_i → alpha = +1
    if a_min > C::zero() {
        Some(-C::one())
    } else if a_max < C::zero() {
        Some(C::one())
    } else {
        None
    }
}

/// Detect unconditionally coupled pairs in a SPIN instance.
///
/// Uses the decomposition
/// `f = A(s_rest) * s_i*s_j + B(s_rest) * s_i + C(s_rest) * s_j + D(s_rest)`
/// and identifies pairs where:
/// - `B ≡ 0`
/// - `C ≡ 0`
/// - `A` has globally constant nonzero sign
///
/// Returns `(i, j, alpha)` with `alpha ∈ {+1, -1}` indicating
/// `s_j = alpha * s_i`.
pub fn detect_uncoupled_pairs<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<(usize, usize, C)> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let n = instance.n_vars();
    let mut in_pair = vec![false; n];
    let mut result = Vec::new();

    'outer: for i in 0..n {
        if in_pair[i] {
            continue;
        }
        'inner: for j in (i + 1)..n {
            if in_pair[j] {
                continue;
            }

            let bounds = pair_bounds(instance, i, j);
            if !bounds.has_a || bounds.has_b || bounds.has_c {
                continue 'inner;
            }

            let Some(alpha) = alpha_from_bounds(bounds.a_min, bounds.a_max) else {
                continue 'inner;
            };

            result.push((i, j, alpha));
            in_pair[i] = true;
            in_pair[j] = true;
            continue 'outer;
        }
    }

    result
}

pub fn detect_uncoupled_pairs_active<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    term_status: &[Option<PartiallyAssignedTerm<C>>],
    current_fixes: &Fixes,
) -> Vec<(usize, usize, C)> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let n = instance.n_vars();
    let mut in_pair = vec![false; n];
    let mut result = Vec::new();

    'outer: for i in current_fixes.assigned.zeroes() {
        if in_pair[i] {
            continue;
        }
        'inner: for j in current_fixes.assigned.zeroes().filter(|&j| j > i) {
            if in_pair[j] {
                continue;
            }

            let bounds = pair_bounds_active(instance, term_status, i, j);
            if !bounds.has_a || bounds.has_b || bounds.has_c {
                continue 'inner;
            }

            let Some(alpha) = alpha_from_bounds(bounds.a_min, bounds.a_max) else {
                continue 'inner;
            };

            result.push((i, j, alpha));
            in_pair[i] = true;
            in_pair[j] = true;
            continue 'outer;
        }
    }

    result
}

/// Detect conditionally coupled pairs in a SPIN instance.
///
/// A pair is reported when `A` has globally constant nonzero sign, exactly one
/// of `B`/`C` is structurally zero, and at least one term contributes to the
/// nonzero side. This captures one-directional coupling relations.
pub fn detect_conditionally_coupled_pairs<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<ConditionalCoupling<C>> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..instance.n_vars() {
        for j in (i + 1)..instance.n_vars() {
            let bounds = pair_bounds(instance, i, j);
            if !bounds.has_a {
                continue;
            }
            let Some(alpha) = alpha_from_bounds(bounds.a_min, bounds.a_max) else {
                continue;
            };

            if !bounds.has_b && bounds.has_c {
                out.push(ConditionalCoupling {
                    i,
                    j,
                    alpha,
                    kind: ConditionalCouplingKind::IGivenJ,
                });
            } else if bounds.has_b && !bounds.has_c {
                out.push(ConditionalCoupling {
                    i,
                    j,
                    alpha,
                    kind: ConditionalCouplingKind::JGivenI,
                });
            }
        }
    }

    out
}

pub fn detect_conditionally_coupled_pairs_active<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    term_status: &[Option<PartiallyAssignedTerm<C>>],
    current_fixes: &Fixes,
) -> Vec<ConditionalCoupling<C>> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in current_fixes.assigned.zeroes() {
        for j in current_fixes.assigned.zeroes().filter(|&j| j > i) {
            let bounds = pair_bounds_active(instance, term_status, i, j);
            if !bounds.has_a {
                continue;
            }
            let Some(alpha) = alpha_from_bounds(bounds.a_min, bounds.a_max) else {
                continue;
            };

            if !bounds.has_b && bounds.has_c {
                out.push(ConditionalCoupling {
                    i,
                    j,
                    alpha,
                    kind: ConditionalCouplingKind::IGivenJ,
                });
            } else if bounds.has_b && !bounds.has_c {
                out.push(ConditionalCoupling {
                    i,
                    j,
                    alpha,
                    kind: ConditionalCouplingKind::JGivenI,
                });
            }
        }
    }

    out
}

/// Detect frustrated spin pairs where coupling sign is context-dependent.
///
/// Reports `(i, j)` when `B ≡ 0`, `C ≡ 0`, and the interval for `A` straddles
/// zero (`A_min < 0 < A_max`). In that case the preferred pair relation can
/// flip depending on the remaining variables.
pub fn detect_frustrated_pairs<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<(usize, usize)> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let mut out = Vec::new();
    for i in 0..instance.n_vars() {
        for j in (i + 1)..instance.n_vars() {
            let bounds = pair_bounds(instance, i, j);
            if !bounds.has_a || bounds.has_b || bounds.has_c {
                continue;
            }
            if bounds.a_min < C::zero() && bounds.a_max > C::zero() {
                out.push((i, j));
            }
        }
    }
    out
}

/// Compute the variable fixes that implement a spin coupling substitution.
///
/// For a coupled pair `(i, j, α)`, substitution `s_j = α * s_i` followed by
/// fixing `s_i = +1` yields fixes `(i, +1)` and `(j, α)`.
pub fn coupling_fixes<C: Coeff>(pairs: &[(usize, usize, C)]) -> Vec<(usize, C)> {
    let mut fixes = Vec::with_capacity(pairs.len() * 2);
    for &(i, j, alpha) in pairs {
        fixes.push((i, C::one()));
        fixes.push((j, alpha));
    }
    fixes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HuboModel;

    #[test]
    fn unconditional_detects_pure_pair() {
        // f = 3·s₀·s₁; A=3>0 → optimal s₀·s₁=-1 → alpha=-1 (s₁=-s₀)
        let inst = HuboModel::spin(2).add_term(&[0, 1], 3.0).build();
        let coupled = detect_uncoupled_pairs(&inst);
        assert_eq!(coupled, vec![(0, 1, -1.0)]);
        // Verify the fixes achieve the minimum: s₀=+1, s₁=-1 → f = 3·(+1)·(−1) = −3
        let fixes = coupling_fixes(&coupled);
        assert_eq!(fixes, vec![(0, 1.0), (1, -1.0)]);
    }

    #[test]
    fn unconditional_not_detected_when_solo_term_exists() {
        let inst = HuboModel::spin(2)
            .add_linear(0, 1.0)
            .add_term(&[0, 1], 3.0)
            .build();
        assert!(detect_uncoupled_pairs(&inst).is_empty());
    }

    #[test]
    fn unconditional_negative_alpha_detected() {
        // f = -2·s₀·s₁; A=-2<0 → optimal s₀·s₁=+1 → alpha=+1 (s₁=+s₀)
        let inst = HuboModel::spin(2).add_term(&[0, 1], -2.0).build();
        assert_eq!(detect_uncoupled_pairs(&inst), vec![(0, 1, 1.0)]);
        let fixes = coupling_fixes(&[(0, 1, 1.0f64)]);
        assert_eq!(fixes, vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn unconditional_mixed_signs_not_detected() {
        let inst = HuboModel::spin(4)
            .add_term(&[0, 1], 1.0)
            .add_term(&[0, 1, 2], 2.0)
            .add_term(&[0, 1, 3], -1.0)
            .build();
        assert!(detect_uncoupled_pairs(&inst).is_empty());
    }

    #[test]
    fn unconditional_positive_coeffs_may_still_be_nonconstant() {
        let inst = HuboModel::spin(4)
            .add_term(&[0, 1, 2], 1.0)
            .add_term(&[0, 1, 3], 1.0)
            .build();
        assert!(detect_uncoupled_pairs(&inst).is_empty());
    }

    #[test]
    fn unconditional_detected_when_bound_forces_sign() {
        // A = 3 + s₂ ∈ [2,4] > 0 → alpha = -1 (s₁ = -s₀)
        let inst = HuboModel::spin(3)
            .add_term(&[0, 1], 3.0)
            .add_term(&[0, 1, 2], 1.0)
            .build();
        assert_eq!(detect_uncoupled_pairs(&inst), vec![(0, 1, -1.0)]);
    }

    #[test]
    fn conditional_detects_one_sided_relation_j_given_i() {
        // B != 0 via term [0], C == 0, A > 0 forced.
        let inst = HuboModel::spin(3)
            .add_linear(0, 1.0)
            .add_term(&[0, 1], 3.0)
            .add_term(&[0, 1, 2], 1.0)
            .build();

        let cond = detect_conditionally_coupled_pairs(&inst);
        assert_eq!(cond.len(), 1);
        assert_eq!(cond[0].i, 0);
        assert_eq!(cond[0].j, 1);
        assert_eq!(cond[0].alpha, -1.0);
        assert_eq!(cond[0].kind, ConditionalCouplingKind::JGivenI);
    }

    #[test]
    fn conditional_detects_one_sided_relation_i_given_j() {
        // B == 0, C != 0 via term [1], A > 0 forced.
        let inst = HuboModel::spin(3)
            .add_linear(1, 1.0)
            .add_term(&[0, 1], 3.0)
            .add_term(&[0, 1, 2], 1.0)
            .build();

        let cond = detect_conditionally_coupled_pairs(&inst);
        assert_eq!(cond.len(), 1);
        assert_eq!(cond[0].kind, ConditionalCouplingKind::IGivenJ);
    }

    #[test]
    fn frustrated_pair_detected_when_a_straddles_zero() {
        let inst = HuboModel::spin(4)
            .add_term(&[0, 1, 2], 1.0)
            .add_term(&[0, 1, 3], 1.0)
            .build();

        let frustrated = detect_frustrated_pairs(&inst);
        assert_eq!(frustrated, vec![(0, 1)]);
    }

    #[test]
    fn frustrated_not_reported_for_unconditional_pair() {
        let inst = HuboModel::spin(2).add_term(&[0, 1], 3.0).build();
        assert!(detect_frustrated_pairs(&inst).is_empty());
    }
}
