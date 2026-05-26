use crate::fixes::Fixes;

use super::*;

// ── LexComparisonConstraint ────────────────────────────────────────────────

/// Lex-comparison constraint enforcing `a ≤_lex p(a)` for a generator permutation `p`.
///
/// `pairs` contains `(k, p⁻¹[k])` for every non-fixed position, in index order.
/// Propagation scans pairs from the start and, at the first unresolved pair:
/// - `(true, false)`  → infeasible (a[k] > a[p⁻¹[k]]).
/// - `(false, true)`  → satisfied; stop.
/// - `(true, free)`   → fix free to `true` (false would immediately violate).
/// - `(free, false)`  → fix free to `false` (true would immediately violate).
/// - Equal or one-free-safe → continue to next pair.
///
/// This is always sound for any permutation symmetry group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LexComparisonConstraint {
    pub(crate) pairs: Vec<(usize, usize)>,
}

impl Constraint for LexComparisonConstraint {
    fn key(&self) -> ConstraintKey {
        ConstraintKey::LexComparison {
            pairs: self.pairs.clone(),
        }
    }

    fn cleanup_and_check(&self, fixed: &Fixes) -> ConstraintCleanup {
        for &(i, j) in &self.pairs {
            if fixed.get(i).is_none() || fixed.get(j).is_none() {
                return ConstraintCleanup::Keep;
            }

            let vi = fixed.get(i).unwrap();
            let vj = fixed.get(j).unwrap();

            match (vi, vj) {
                (false, true) => return ConstraintCleanup::Drop,
                (true, false) => return ConstraintCleanup::Infeasible,
                _ => {}
            }
        }
        ConstraintCleanup::Drop
    }

    fn accumulate_branch_scores(&self, fixed: &Fixes, scores: &mut [u64]) {
        for &(i, j) in &self.pairs {
            if fixed.get(i).is_none() {
                scores[i] += 1;
            }
            if fixed.get(j).is_none() {
                scores[j] += 1;
            }
        }
    }

    fn propagate(&self, fixed: &Fixes, fixed_this_round: &mut Fixes) -> ConstraintPropagation {
        let mut changed = false;

        for &(i, j) in &self.pairs {
            let vi = fixed.get(i).or_else(|| fixed_this_round.get(i));
            let vj = fixed.get(j).or_else(|| fixed_this_round.get(j));

            match (vi, vj) {
                (Some(false), Some(true)) => {
                    // a[i] < a[j]: constraint satisfied. Stop.
                    return if changed {
                        ConstraintPropagation::Fixed
                    } else {
                        ConstraintPropagation::NoChange
                    };
                }
                (Some(true), Some(false)) => {
                    return ConstraintPropagation::Infeasible;
                }
                (Some(a), Some(b)) if a == b => {
                    // Equal: continue to next pair.
                    continue;
                }
                (Some(true), None) => {
                    // a[i]=true, a[j]=free: false would give (true,false)=infeasible, so fix to true.
                    if !apply_fix(fixed_this_round, j, true) {
                        return ConstraintPropagation::Infeasible;
                    }
                    changed = true;
                    continue;
                }
                (None, Some(false)) => {
                    // a[j]=false, a[i]=free: true would give (true,false)=infeasible, so fix to false.
                    if !apply_fix(fixed_this_round, i, false) {
                        return ConstraintPropagation::Infeasible;
                    }
                    changed = true;
                    continue;
                }
                _ => {
                    // First genuinely unresolved pair — cannot deduce anything yet.
                    break;
                }
            }
        }

        if changed {
            ConstraintPropagation::Fixed
        } else {
            ConstraintPropagation::NoChange
        }
    }
}

/// A lex-order chain over exchangeable variables from one permutation orbit.
///
/// Enforces `x[vars[0]] <= x[vars[1]] <= ... <= x[vars[k-1]]`.
///
/// For BIN variables this is `0 <= 1`; for SPIN, with internal bool encoding
/// `false=-1, true=+1`, the same monotone bool order matches `-1 <= +1`.
#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LexOrderConstraint {
    pub(crate) vars: Vec<usize>,
}

impl Constraint for LexOrderConstraint {
    fn key(&self) -> ConstraintKey {
        ConstraintKey::LexOrder {
            vars: self.vars.clone(),
        }
    }

    fn cleanup_and_check(&self, fixed: &Fixes) -> ConstraintCleanup {
        let mut all_edges_resolved = true;
        for pair in self.vars.windows(2) {
            let left = pair[0];
            let right = pair[1];

            if fixed.get(left).is_some() && fixed.get(right).is_some() {
                let left_high = fixed.get(left).unwrap();
                let right_high = fixed.get(right).unwrap();

                if left_high && !right_high {
                    return ConstraintCleanup::Infeasible;
                }
            } else {
                all_edges_resolved = false;
            }
        }
        if all_edges_resolved {
            ConstraintCleanup::Drop
        } else {
            ConstraintCleanup::Keep
        }
    }

    fn accumulate_branch_scores(&self, fixed: &Fixes, scores: &mut [u64]) {
        for pair in self.vars.windows(2) {
            let left = pair[0];
            let right = pair[1];

            if fixed.get(left).is_none() {
                scores[left] += 1;
            }
            if fixed.get(right).is_none() {
                scores[right] += 1;
            }
        }
    }

    fn propagate(&self, fixed: &Fixes, fixed_this_round: &mut Fixes) -> ConstraintPropagation {
        let mut changed = false;
        for pair in self.vars.windows(2) {
            let left = pair[0];
            let right = pair[1];

            let left_val = match fixed.get(left) {
                Some(fixed_val) => Some(fixed_val),
                None => fixed_this_round.get(left),
            };

            let right_val = match fixed.get(right) {
                Some(fixed_val) => Some(fixed_val),
                None => fixed_this_round.get(right),
            };

            match (left_val, right_val) {
                (Some(true), Some(false)) => {
                    return ConstraintPropagation::Infeasible;
                }
                (Some(true), None) => {
                    if !apply_fix(fixed_this_round, right, true) {
                        return ConstraintPropagation::Infeasible;
                    }
                    changed = true;
                }
                (None, Some(false)) => {
                    if !apply_fix(fixed_this_round, left, false) {
                        return ConstraintPropagation::Infeasible;
                    }
                    changed = true;
                }
                _ => {}
            }
        }

        if changed {
            // log::info!("Propagated lex-order constraint: {:?}", self.vars);
            ConstraintPropagation::Fixed
        } else {
            ConstraintPropagation::NoChange
        }
    }
}
