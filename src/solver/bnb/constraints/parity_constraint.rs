use crate::fixes::Fixes;

use super::*;

/// A parity constraint derived from a term whose product value is forced by
/// the incumbent.
///
/// For **spin** variables: the product of the `free_vars` must have an odd
/// (`odd_required = true`) or even (`odd_required = false`) number of -1
/// assignments, equivalently the product of those variables must equal -1 or +1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParityConstraint {
    /// The free variables participating in the constraint (indices into the instance).
    pub(crate) free_vars: Vec<usize>,
    /// `true`  -> odd number of -1s required among `free_vars` (product = -1 given even assigned sign).
    /// `false` -> even number of -1s required (product = +1).
    pub(crate) odd_required: bool,
}

impl Constraint for ParityConstraint {
    fn key(&self) -> ConstraintKey {
        ConstraintKey::Parity {
            free_vars: self.free_vars.clone(),
            odd_required: self.odd_required,
        }
    }

    fn cleanup_and_check(&self, fixed: &Fixes) -> ConstraintCleanup {
        let mut neg_fixed: u32 = 0;
        let mut remaining = 0usize;
        for &v in &self.free_vars {
            if let Some(high) = fixed.get(v) {
                if !high {
                    neg_fixed += 1;
                }
            } else {
                remaining += 1;
            }
        }
        if remaining == 0 {
            if (neg_fixed % 2 == 1) != self.odd_required {
                return ConstraintCleanup::Infeasible;
            }
            return ConstraintCleanup::Drop;
        }
        ConstraintCleanup::Keep
    }

    fn accumulate_branch_scores(&self, fixed: &Fixes, scores: &mut [u64]) {
        let mut free_count = 0usize;

        for &v in &self.free_vars {
            if !fixed.assigned.contains(v) {
                free_count += 1;
            }
        }
        if free_count < 2 {
            return;
        }
        for &v in &self.free_vars {
            if !fixed.assigned.contains(v) {
                scores[v] += 1;
            }
        }
    }

    fn propagate(&self, fixed: &Fixes, fixed_this_round: &mut Fixes) -> ConstraintPropagation {
        let mut neg_fixed: u32 = 0;
        let mut remaining_free: Vec<usize> = Vec::with_capacity(self.free_vars.len());

        for &v in &self.free_vars {
            if let Some(high) = fixed.get(v) {
                if !high {
                    neg_fixed += 1;
                }
                continue;
            }

            match fixed_this_round.get(v) {
                Some(high) => {
                    if !high {
                        neg_fixed += 1;
                    }
                }
                None => remaining_free.push(v),
            }
        }

        if remaining_free.is_empty() {
            if (neg_fixed % 2 == 1) != self.odd_required {
                return ConstraintPropagation::Infeasible;
            }
            return ConstraintPropagation::NoChange;
        }

        if remaining_free.len() == 1 {
            let xi = remaining_free[0];
            let xi_must_be_neg = self.odd_required != (neg_fixed % 2 == 1);
            let high = !xi_must_be_neg;

            if !apply_fix(fixed_this_round, xi, high) {
                return ConstraintPropagation::Infeasible;
            }
            return ConstraintPropagation::Fixed;
        }

        ConstraintPropagation::NoChange
    }
}
