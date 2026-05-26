use crate::{
    Coeff,
    domain::{VarDomain, VarType},
    fixes::Fixes,
    instance::HuboInstance,
    solver::bnb::{Node, PartiallyAssignedTerm},
};

/// Variable fixing using the lower-bound gap (the dual of incumbent-based propagation).
///
/// For each free variable `v`, compute the exact increase in the cheap base lower
/// bound when fixing `v` to each value.  If that increase meets the incumbent, the
/// other value is forced.  Returns `None` if the node is infeasible (both values of
/// the same variable are pruned).
///
/// BIN:  delta(v→1) = Σ{ coeff  | linear-in-free-vars terms with v, coeff > 0 }
///       delta(v→0) = Σ{ |coeff| | active terms with v, coeff < 0 }
///
/// SPIN: delta(v→+1) = Σ{ 2·coeff  | linear-in-free-vars terms with v, coeff > 0 }
///       delta(v→−1) = Σ{ 2·|coeff| | linear-in-free-vars terms with v, coeff < 0 }
///
/// The base cheap bound (without the parity bonus) is computed internally so that
/// `base_lb + delta` equals the child's base cheap bound exactly.
pub fn lb_fixing<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    incumbent: C,
) -> Option<Vec<(usize, C)>> {
    let lb_f = crate::bounds::cheap::lower_bound_base(instance, node).to_f64();
    let ub_f = incumbent.to_f64();
    let mut fixes: Vec<(usize, C)> = Vec::new();

    for var in node.fixed.iter_unassigned() {
        let (delta_high, delta_low) = lb_deltas::<C, V>(instance, node, var);

        let high_pruned = lb_f + delta_high >= ub_f;
        let low_pruned  = lb_f + delta_low  >= ub_f;

        if high_pruned && low_pruned {
            return None; // infeasible
        }
        if high_pruned {
            fixes.push((var, if V::VAR_TYPE == VarType::Bin { C::zero() } else { -C::one() }));
        } else if low_pruned {
            fixes.push((var, C::one()));
        }
    }
    Some(fixes)
}

#[inline]
fn lb_deltas<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    var: usize,
) -> (f64, f64) {
    let mut delta_high = 0.0f64;
    let mut delta_low  = 0.0f64;

    for &ti in &instance.var_terms[var] {
        let Some(status) = &node.term_status[ti] else { continue };
        let c = status.coeff.to_f64();
        if c == 0.0 { continue; }

        match V::VAR_TYPE {
            VarType::Bin => {
                if status.free_variables.len() == 1 {
                    if c > 0.0 { delta_high += c; }
                    if c < 0.0 { delta_low  -= c; }
                } else if c < 0.0 {
                    delta_low -= c; // killing a negative term raises LB
                }
            }
            VarType::Spin => {
                // Only linear terms matter; higher-order: delta = 0 for both directions.
                if status.free_variables.len() == 1 {
                    if c > 0.0 { delta_high += 2.0 * c; }
                    if c < 0.0 { delta_low  -= 2.0 * c; }
                }
            }
        }
    }
    (delta_high, delta_low)
}

pub fn dominance_fixes<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> Vec<(usize, C)> {
    match V::VAR_TYPE {
        VarType::Bin => binary_dominance_fixes(instance, &node.term_status, &node.fixed),
        VarType::Spin => spin_dominance_fixes(instance, &node.term_status, &node.fixed),
    }
}

/// Binary dominance on the active residual problem described by `term_status`.
pub fn binary_dominance_fixes<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    term_status: &[Option<PartiallyAssignedTerm<C>>],
    current_fixes: &Fixes,
) -> Vec<(usize, C)> {
    let mut fixes = Vec::new();

    for var in current_fixes.iter_unassigned() {
        let mut sum_neg = C::zero();
        let mut sum_pos = C::zero();
        let mut sum_const = C::zero();
        let mut appears = false;

        for &term_idx in &instance.var_terms[var] {
            let Some(status) = &term_status[term_idx] else {
                continue;
            };
            if status.coeff == C::zero() {
                continue;
            }

            appears = true;
            let coeff = status.coeff;
            if status.free_variables.len() == 1 {
                sum_const += coeff;
            } else if coeff >= C::zero() {
                sum_pos += coeff;
            } else {
                sum_neg += coeff;
            }
        }

        if !appears {
            continue;
        }

        if sum_const + sum_neg >= C::zero() {
            fixes.push((var, C::zero()));
        } else if sum_const + sum_pos <= C::zero() {
            fixes.push((var, C::one()));
        }
    }

    fixes
}

/// Spin dominance on the active residual problem described by `term_status`.
pub fn spin_dominance_fixes<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    term_status: &[Option<PartiallyAssignedTerm<C>>],
    current_fixes: &Fixes,
) -> Vec<(usize, C)> {
    let mut fixes = Vec::new();

    for var in current_fixes.iter_unassigned() {
        let mut sum_other = C::zero();
        let mut sum_const = C::zero();
        let mut appears = false;

        for &term_idx in &instance.var_terms[var] {
            let Some(status) = &term_status[term_idx] else {
                continue;
            };

            let coeff = status.coeff;
            if coeff == C::zero() {
                continue;
            }

            appears = true;

            if status.free_variables.len() == 1 {
                sum_const += coeff;
            } else {
                sum_other += coeff.abs();
            }
        }

        if !appears {
            continue;
        }

        if sum_const.abs() >= sum_other && sum_const != C::zero() {
            fixes.push((var, sum_const.signum() * -C::one()));
        }
    }

    fixes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::{
        fixes::Fixes,
        kernelization::Kernelizer,
        model::HuboModel,
        solver::bnb::{ConstraintHandler, Node, PartiallyAssignedTerm},
    };
    use std::sync::Arc;

    #[test]
    fn binary_dominance_fixes_variable() {
        let instance = HuboModel::binary(2)
            .add_linear(0, 5.0)
            .add_term(&[0, 1], 2.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::default();
        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();

        assert_eq!(node.fixed.get(0), Some(false));
        assert!(report.rule_fixed >= 1);
    }

    #[test]
    fn spin_dominance_fixes_variable() {
        let instance = HuboModel::spin(2)
            .add_linear(0, 3.0)
            .add_term(&[0, 1], 1.0)
            .build();
        let term_status: Vec<_> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();
        let fixed = Fixes::new(instance.n_vars());
        let mut node = Node {
            fixed,
            lb: 0.0,
            offset: 0.0,
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        };

        let kernelizer = Kernelizer::default();
        let report = kernelizer
            .kernelize(&Arc::new(instance), &mut node, None)
            .unwrap();

        assert_eq!(node.fixed.get(0), Some(false));
        assert!(report.rule_fixed >= 1);
    }
}
