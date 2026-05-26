use crate::fixes::Fixes;

use super::*;

/// An incumbent-based knapsack cover cut (Section 4.3.1 / Theorem 4.3).
///
/// For a binary instance with incumbent `U`, split the terms into
///   S1 = {high-coefficient terms we will track} and S2 = rest.
///
/// From `f(x) <= U` and a lower bound `f2_lb` on S2:
///
///   sum_{S in S1} c_S * T_S  <=  U - f2_lb   (T_S = prod_{i in S} x_i in {0,1})
///
/// A *cover* C subseteq S1 satisfies sum_{S in C} c_S > U - f2_lb.
/// Any feasible assignment must violate at least one item in C, giving:
///
///   sum_{S in C} T_S  <=  |C| - 1
///
/// i.e. at most `max_active = |C| - 1` items can be simultaneously active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverConstraint {
    /// Variable index sets for each cover item (one per selected term).
    pub(crate) items: Vec<Vec<usize>>,
    /// Maximum number of simultaneously active items: |C| - 1.
    pub(crate) max_active: usize,
}

impl Constraint for CoverConstraint {
    fn key(&self) -> ConstraintKey {
        ConstraintKey::Cover {
            items: self.items.clone(),
            max_active: self.max_active,
        }
    }

    fn cleanup_and_check(&self, fixed: &Fixes) -> ConstraintCleanup {
        let mut active = 0usize;
        let mut potential = 0usize;
        for item in &self.items {
            let mut all_one = true;
            let mut any_zero = false;
            for &v in item {
                if let Some(high) = fixed.get(v) {
                    if !high {
                        any_zero = true;
                        break;
                    }
                } else {
                    all_one = false;
                }
            }
            if any_zero {
            } else if all_one {
                active += 1;
            } else {
                potential += 1;
            }
        }
        if active > self.max_active {
            return ConstraintCleanup::Infeasible;
        }
        if active + potential <= self.max_active {
            return ConstraintCleanup::Drop;
        }
        ConstraintCleanup::Keep
    }

    fn accumulate_branch_scores(&self, fixed: &Fixes, scores: &mut [u64]) {
        for item in &self.items {
            let any_zero = item.iter().any(|&v| fixed.get(v) == Some(false));
            if any_zero {
                continue;
            }
            for &v in item {
                if !fixed.assigned.contains(v) {
                    scores[v] += 1;
                }
            }
        }
    }

    fn propagate(&self, fixed: &Fixes, fixed_this_round: &mut Fixes) -> ConstraintPropagation {
        let mut active = 0usize;
        let mut potential_items: Vec<&Vec<usize>> = Vec::new();

        for item in &self.items {
            let mut all_one = true;
            let mut any_zero = false;
            for &v in item {
                let val = match fixed.get(v) {
                    Some(fixed_val) => Some(fixed_val),
                    None => fixed_this_round.get(v),
                };

                match val {
                    Some(false) => {
                        any_zero = true;
                        break;
                    }
                    Some(true) => {}
                    None => {
                        all_one = false;
                    }
                }
            }
            if any_zero {
            } else if all_one {
                active += 1;
            } else {
                potential_items.push(item);
            }
        }

        if active > self.max_active {
            return ConstraintPropagation::Infeasible;
        }

        let mut changed = false;
        if active == self.max_active {
            for item in potential_items {
                let mut free: Vec<usize> = Vec::new();
                for &v in item {
                    let val = match fixed.get(v) {
                        Some(fixed_val) => Some(fixed_val),
                        None => fixed_this_round.get(v),
                    };
                    if val.is_none() {
                        free.push(v);
                    }
                }
                if free.is_empty() {
                    return ConstraintPropagation::Infeasible;
                }
                if free.len() == 1 {
                    if !apply_fix(fixed_this_round, free[0], false) {
                        return ConstraintPropagation::Infeasible;
                    }
                    changed = true;
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
