use fixedbitset::FixedBitSet;
use std::collections::HashMap;

use crate::util::bitset::BitSet;
use crate::{
    coeff::Coeff,
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

use super::Node;

#[derive(Debug, Clone, Copy)]
pub struct Cheap;

impl Default for Cheap {
    fn default() -> Self {
        Cheap
    }
}

/// The decomposable base bound (no unsat-core bonus) used by `lb_fixing`.
/// For BIN this equals `lower_bound`; for SPIN it equals `lower_bound_spin_base`.
pub(crate) fn lower_bound_base<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> C {
    if V::VAR_TYPE == VarType::Bin {
        lower_bound_bin(instance, node)
    } else {
        lower_bound_spin_base(instance, node)
    }
}

/// Computes the lower bound on the objective over the given node's sub-tree.
pub(crate) fn lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> C {
    if V::VAR_TYPE == VarType::Bin {
        lower_bound_bin(instance, node)
    } else {
        lower_bound_spin(instance, node)
    }
}

fn lower_bound_bin<C: Coeff, V: VarDomain>(instance: &HuboInstance<C, V>, node: &Node<C>) -> C {
    let mut lb = instance.offset + node.offset;

    for term_status in node.term_status.iter().flat_map(|x| x.as_ref()) {
        lb += if term_status.free_variables.is_empty() {
            // Fully assigned
            term_status.coeff
        } else {
            // Partially assigned
            if term_status.coeff >= C::zero() {
                C::zero()
            } else {
                term_status.coeff
            }
        };
    }

    lb
}

fn lower_bound_spin<C: Coeff, V: VarDomain>(instance: &HuboInstance<C, V>, node: &Node<C>) -> C {
    let lb = lower_bound_spin_base(instance, node);
    let core_bonus = unsat_core_bonus(instance, node);

    lb + core_bonus
}

// Same as lower_bound_bin except with the spin-specific term contributions (flipping sign for negative coeffs when still active).
pub(crate) fn lower_bound_spin_base<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> C {
    let mut lb = instance.offset + node.offset;

    for term_status in node.term_status.iter().flat_map(|x| x.as_ref()) {
        lb += if term_status.free_variables.is_empty() {
            // Fully assigned
            term_status.coeff
        } else {
            // Partially assigned
            -term_status.coeff.abs()
        };
    }

    lb
}

/// Detect one inconsistent parity core among active equations.
///
/// Returns the equation indices in an odd RHS contradiction (xor of selected
/// equations gives `0 = 1`) if one exists.
///
/// Rows are represented as `BitSet` so that each XOR step is a
/// word-level operation rather than a sorted-vector merge.
pub(crate) fn parity_unsat_core(eqns: Vec<BitSet>, n_vars: usize) -> Option<Vec<usize>> {
    let n_eqs = eqns.len();
    let mut basis: Vec<Option<(BitSet, BitSet)>> = vec![None; n_vars];

    'outer: for (eq_idx, eq) in eqns.into_iter().enumerate() {
        let mut var_and_rhs = eq;
        let mut sup_bits = BitSet::new(n_eqs);
        sup_bits.set(eq_idx, true);

        loop {
            let Some(pivot) = var_and_rhs.iter_ones().take_while(|&i| i < n_vars).next() else {
                break;
            };
            if let Some((bvnr, bs)) = &basis[pivot] {
                var_and_rhs ^= bvnr;
                sup_bits ^= bs;
            } else {
                basis[pivot] = Some((var_and_rhs, sup_bits));
                continue 'outer;
            }
        }

        if var_and_rhs.get(n_vars) {
            return Some(sup_bits.iter_ones().collect());
        }
    }

    None
}

pub fn get_residual_penalties<C: Coeff>(node: &Node<C>) -> (Vec<C>, Vec<usize>) {
    let mut residual_penalties = Vec::new();
    let mut local_to_global = Vec::new();

    // set residual weights to 2·|coeff| of each active term
    for (term_idx, term_status) in node
        .term_status
        .iter()
        .enumerate()
        .flat_map(|(i, x)| x.as_ref().map(|v| (i, v)))
    {
        if term_status.free_variables.is_empty() || term_status.coeff == C::zero() {
            continue;
        }
        let twice = term_status.coeff + term_status.coeff;
        residual_penalties.push(twice.abs());
        local_to_global.push(term_idx);
    }

    (residual_penalties, local_to_global)
}

/// Check whether `hint` (global term indices) is still a valid parity contradiction
/// in the current active equation set.  Returns the positions of the hint terms in
/// `active_eqs` if the XOR of their equations gives the empty-variable / odd-RHS
/// contradiction, `None` otherwise.
fn validate_hint_core(
    hint: &[usize],
    global_to_local: &HashMap<usize, usize>,
    local_to_pos: &HashMap<usize, usize>,
    active_eqs: &[BitSet],
    n_vars: usize,
) -> Option<Vec<usize>> {
    if hint.len() < 2 {
        return None;
    }

    let mut xor_check = BitSet::new(n_vars + 1);
    let mut positions = Vec::with_capacity(hint.len());

    for &ti in hint {
        let li = global_to_local.get(&ti)?;
        let &pos = local_to_pos.get(li)?;
        xor_check ^= &active_eqs[pos];
        positions.push(pos);
    }

    // Valid contradiction: no variable bits set, RHS bit (at n_vars) is 1.
    if xor_check.get(n_vars)
        && xor_check
            .iter_ones()
            .take_while(|&i| i < n_vars)
            .next()
            .is_none()
    {
        Some(positions)
    } else {
        None
    }
}

/// Build a knapsack-style LB tightening from incompatible preferred parities
/// across multiple spin terms.
///
/// Returns `(bonus, first_core)` where `first_core` holds the global term indices
/// of the first parity-unsat core found (used for branching heuristics).
///
/// `hint_core` carries the parent node's first core as a warm-start: it is validated
/// cheaply (O(k) XOR ops) before the first Gaussian-elimination pass, and used
/// directly if still consistent, saving O(n·m) GE work.
fn unsat_core_bonus_and_core<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    hint_core: Option<&[usize]>,
) -> (C, Option<Vec<usize>>) {
    let (mut residual_penalties, local_to_global) = get_residual_penalties(node);

    if residual_penalties.len() < 2 {
        return (C::zero(), None);
    }

    // Reverse map used for hint validation: global term idx → local idx.
    let global_to_local: HashMap<usize, usize> = local_to_global
        .iter()
        .enumerate()
        .map(|(li, &gi)| (gi, li))
        .collect();

    let mut active_terms = FixedBitSet::with_capacity(residual_penalties.len());
    active_terms.set_range(.., true);

    let mut bonus = C::zero();
    let mut first_core: Option<Vec<usize>> = None;
    let mut is_first_iter = true;

    loop {
        let mut active_eqs: Vec<BitSet> = Vec::with_capacity(residual_penalties.len());
        let mut active_map: Vec<usize> = Vec::with_capacity(residual_penalties.len());
        let mut inactive_terms = Vec::new();

        // On the first iteration we may need a local_idx → active_eqs position map for
        // hint validation.  Build it only when a hint is available.
        let mut local_to_pos: HashMap<usize, usize> = if is_first_iter && hint_core.is_some() {
            HashMap::with_capacity(residual_penalties.len())
        } else {
            HashMap::new()
        };

        for local_term_idx in active_terms.ones() {
            let term_idx = local_to_global[local_term_idx];

            if residual_penalties[local_term_idx] <= C::zero() {
                inactive_terms.push(local_term_idx);
                continue;
            }

            let mut vars_and_rhs = BitSet::new(instance.n_vars() + 1);
            for &v in &instance.terms[term_idx].indices {
                vars_and_rhs.set(v, true);
            }
            vars_and_rhs.set(
                instance.n_vars(),
                node.term_status[term_idx].as_ref().unwrap().coeff > C::zero(),
            );

            if is_first_iter && hint_core.is_some() {
                local_to_pos.insert(local_term_idx, active_map.len());
            }
            active_map.push(local_term_idx);
            active_eqs.push(vars_and_rhs);
        }

        for local_term_idx in inactive_terms {
            active_terms.set(local_term_idx, false);
        }

        if active_eqs.len() < 2 {
            break;
        }

        // On the first iteration, try the inherited hint before running full GE.
        let hint_positions: Option<Vec<usize>> = if is_first_iter {
            hint_core.and_then(|hint| {
                validate_hint_core(
                    hint,
                    &global_to_local,
                    &local_to_pos,
                    &active_eqs,
                    instance.n_vars(),
                )
            })
        } else {
            None
        };

        is_first_iter = false;

        let core_local: Vec<usize> = if let Some(positions) = hint_positions {
            positions
        } else {
            match parity_unsat_core(active_eqs, instance.n_vars()) {
                Some(c) => c,
                None => break,
            }
        };

        let mut min_pen = C::max_value();
        let mut min_pen_idx = None;
        let mut has_penalty = false;
        let mut core_local_indices: Vec<usize> = Vec::with_capacity(core_local.len());

        for local_idx in core_local {
            let gi = active_map[local_idx];
            core_local_indices.push(gi);

            let p = residual_penalties[gi];
            if !has_penalty || p < min_pen {
                min_pen_idx = Some(gi);
                min_pen = p;
                has_penalty = true;
            }
        }

        if !has_penalty || min_pen <= C::zero() {
            break;
        }

        // Capture global term indices of the first core for branching reuse.
        if first_core.is_none() {
            first_core = Some(
                core_local_indices
                    .iter()
                    .map(|&li| local_to_global[li])
                    .collect(),
            );
        }

        bonus += min_pen;
        for gi in core_local_indices {
            residual_penalties[gi] -= min_pen;
        }

        if let Some(min_pen_idx) = min_pen_idx {
            active_terms.set(min_pen_idx, false);
        } else {
            unreachable!("has_penalty implies min_pen_idx is Some");
        }
    }

    (bonus, first_core)
}

pub(crate) fn unsat_core_bonus<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> C {
    unsat_core_bonus_and_core(instance, node, None).0
}

/// Like [`lower_bound`] but also returns the first parity core for reuse in
/// branching variable selection, avoiding a redundant Gaussian-elimination pass.
///
/// Reads `node.cheap_core_hint` as a warm-start for the GE pass and writes the
/// freshly computed core back so child nodes can inherit it.
pub(crate) fn compute_with_core<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
) -> (C, Option<Vec<usize>>) {
    if V::VAR_TYPE == VarType::Bin {
        (lower_bound_bin(instance, node), None)
    } else {
        let lb_base = lower_bound_spin_base(instance, node);
        let hint = node
            .lb_warm_start
            .as_ref()
            .and_then(|ws| ws.downcast_ref::<Vec<usize>>())
            .cloned();
        let (bonus, core) = unsat_core_bonus_and_core(instance, node, hint.as_deref());
        if let Some(ref c) = core {
            node.lb_warm_start = Some(std::sync::Arc::new(c.clone()));
        }
        (lb_base + bonus, core)
    }
}

pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
) -> C {
    if V::VAR_TYPE == VarType::Bin {
        lower_bound_bin(instance, node)
    } else {
        compute_with_core(instance, node).0
    }
}
