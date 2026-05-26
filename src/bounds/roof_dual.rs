use std::sync::Arc;

use crate::coeff::Coeff;
use crate::solver::bnb::Node;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
    term::Term,
};

/// Apply roof-duality fixings on the active node when the residual problem has
/// effective order at most 2. Returns `true` if no contradiction is detected.
pub(crate) fn apply_roof_dual_fixings<C: Coeff, V: VarDomain>(
    node: &mut Node<C>,
    instance: &Arc<HuboInstance<C, V>>,
    bin_roof: fn(&HuboInstance<C, V>) -> Vec<(usize, C)>,
    spin_roof: fn(&HuboInstance<C, V>) -> Vec<(usize, C)>,
) -> bool {
    let max_free_order = node
        .term_status
        .iter()
        .filter_map(|s| s.as_ref().map(|t| t.free_variables.len()))
        .max()
        .unwrap_or(0);

    if max_free_order > 2 {
        return true;
    }

    loop {
        let Some((residual, compact_to_source)) = build_residual_quadratic(node, instance) else {
            return true;
        };

        let fixes = match V::VAR_TYPE {
            VarType::Bin => bin_roof(&residual),
            VarType::Spin => spin_roof(&residual),
        };
        if fixes.is_empty() {
            return true;
        }

        let mut applied_any = false;
        for (compact_idx, value) in fixes {
            if compact_idx >= compact_to_source.len() {
                continue;
            }
            let source_idx = compact_to_source[compact_idx];
            let high = value > C::zero();

            if node.fixed.assigned.contains(source_idx) {
                if node.fixed.values.contains(source_idx) != high {
                    node.lb = C::max_value();
                    return false;
                }
                continue;
            }

            if node.set_variable(instance, source_idx, high).is_err() {
                node.lb = C::max_value();
                return false;
            }
            applied_any = true;
        }

        if !applied_any {
            return true;
        }
    }
}

fn build_residual_quadratic<C: Coeff, V: VarDomain>(
    node: &Node<C>,
    instance: &HuboInstance<C, V>,
) -> Option<(HuboInstance<C, V>, Vec<usize>)> {
    // Iterating 0..n_vars in order gives a sorted free_vars without an extra sort.
    let free_vars: Vec<usize> = (0..instance.n_vars())
        .filter(|&v| !node.fixed.assigned.contains(v))
        .collect();
    if free_vars.is_empty() {
        return None;
    }

    // node.offset holds contributions from all terms fixed via set_variable.
    let mut offset = node.offset;
    let mut terms: Vec<Term<C>> = Vec::new();

    for term_status in node.term_status.iter().filter_map(|s| s.as_ref()) {
        let coeff = term_status.coeff;
        if coeff == C::zero() {
            continue;
        }

        let free_variables = &term_status.free_variables;

        if free_variables.is_empty() {
            // Fully resolved term that hasn't been cleaned up yet.
            offset += coeff;
            continue;
        }

        if free_variables.len() > 2 {
            return None;
        }

        // free_vars is sorted, so binary search gives the compact index.
        // free_variables are always distinct variable indices, so no dedup needed.
        let mut mapped: Vec<usize> = free_variables
            .iter()
            .map(|&v| {
                free_vars
                    .binary_search(&v)
                    .expect("active free variable missing from free_vars")
            })
            .collect();
        mapped.sort_unstable();

        terms.push(Term { indices: mapped, coeff });
    }

    Some((HuboInstance::new(free_vars.len(), offset, terms), free_vars))
}
