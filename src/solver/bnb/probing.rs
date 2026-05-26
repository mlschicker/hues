use fixedbitset::FixedBitSet;

use crate::bounds::cheap::lower_bound as cheap_lb;
use crate::fixes::Fixes;

use super::*;

/// Probe free variables at a BnB node to derive forced fixings.
///
/// For each free variable (up to `config.max_vars`), both assignments are
/// explored by temporarily fixing the variable and running constraint
/// propagation.  Three deductions are possible:
///
/// - **One branch infeasible** (lb ≥ incumbent or propagation contradiction):
///   fix the variable to the other value.
/// - **Both branches infeasible**: the node is infeasible; return `false`.
/// - **Both branches agree on another variable's value**: fix it unconditionally
///   (double-implication).
///
/// Only constraint propagation and the cheap bound are run per probe — the
/// full configured oracle is too expensive to call inside probing.
///
/// Returns `false` if the node is proven infeasible, `true` otherwise.
pub(super) fn probe_node<C: Coeff, V: VarDomain>(
    instance: &Arc<HuboInstance<C, V>>,
    node: &mut Node<C>,
    constraint_handler: &ConstraintHandler,
    incumbent: Option<C>,
    config: &ProbingConfig,
) -> bool {
    if !config.enabled || node.fixed.num_free() < config.min_free_vars {
        return true;
    }

    let n_vars = instance.n_vars();
    let mut forced = Fixes::new(n_vars);
    let mut n_probed = 0;

    for var in node.fixed.iter_unassigned() {
        if n_probed >= config.max_vars {
            break;
        }
        if forced.get(var).is_some() {
            continue;
        }
        n_probed += 1;

        let mut feasible = [true, true];
        // Bitsets for the fixings produced by each probe branch; used for
        // O(n/64) implication intersection rather than Vec linear scan.
        let mut probe_assigned: [FixedBitSet; 2] = [
            FixedBitSet::with_capacity(n_vars),
            FixedBitSet::with_capacity(n_vars),
        ];
        let mut probe_values: [FixedBitSet; 2] = [
            FixedBitSet::with_capacity(n_vars),
            FixedBitSet::with_capacity(n_vars),
        ];

        for (idx, &high) in [false, true].iter().enumerate() {
            let mut probe = node.child(instance, var, high);

            if !propagate_constraints_only(instance, constraint_handler, &mut probe) {
                feasible[idx] = false;
                continue;
            }

            // Cheap bound only — the full oracle is too expensive per probe.
            if incumbent.is_some_and(|inc| cheap_lb(instance, &probe) >= inc) {
                feasible[idx] = false;
                continue;
            }

            probe_assigned[idx] = probe.fixed.assigned.clone();
            probe_values[idx] = probe.fixed.values.clone();
        }

        match feasible {
            [false, false] => return false,
            [false, true] => {
                if !apply_fix(&mut forced, var, true) {
                    return false;
                }
            }
            [true, false] => {
                if !apply_fix(&mut forced, var, false) {
                    return false;
                }
            }
            [true, true] => {
                // Variables fixed in both branches to the same value are forced.
                // Compute the intersection in O(n/64) via bitset AND.
                let mut both = probe_assigned[0].clone();
                both.intersect_with(&probe_assigned[1]);
                both.difference_with(&node.fixed.assigned);
                both.set(var, false);

                for v in both.ones() {
                    if forced.get(v).is_some() {
                        continue;
                    }
                    let val = probe_values[0].contains(v);
                    if probe_values[1].contains(v) == val
                        && !apply_fix(&mut forced, v, val) {
                            return false;
                        }
                }
            }
        }
    }

    // Apply all deduced fixings to the node and re-propagate once.
    let mut any_fixed = false;
    for (v, high) in forced.iter_fixed() {
        if node.fixed.get(v).is_none() {
            if node.set_variable(instance, v, high).is_err() {
                return false;
            }
            any_fixed = true;
        }
    }

    if any_fixed {
        if !propagate_constraints_only(instance, constraint_handler, node) {
            return false;
        }
        if incumbent.is_some_and(|inc| node.lb >= inc) {
            return false;
        }
    }

    true
}
