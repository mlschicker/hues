use crate::bounds::cheap::compute as cheap_lb;

use super::*;

/// Result of the branching variable selection step.
pub(super) enum BranchChoice {
    /// Branch on this variable next.
    On(usize),
    /// No free variables remain — this is a leaf node.
    Leaf,
    /// Strong branching proved both children of some candidate infeasible:
    /// the node itself is infeasible and should be pruned.
    Infeasible,
}

/// Select a branching variable, using strong branching when configured.
///
/// **Heuristic fallback** (`strong_branching.enabled = false` or too few free
/// variables): delegates directly to the existing heuristic score
/// (parity-core → constraint activity → first unassigned).
///
/// **Strong branching**: evaluates both children of each of the top
/// `max_candidates` heuristic candidates using constraint propagation and the
/// cheap lower bound, then picks the variable that maximises the *product of
/// bound improvements* `(Δlb₀ + ε) × (Δlb₁ + ε)` — the standard
/// reliability-branching score that balances both branches.
///
/// Early-exit rules applied per candidate:
/// - Both children infeasible/prunable → `BranchChoice::Infeasible` (node is dead).
/// - Exactly one child infeasible      → `BranchChoice::On(var)` immediately
///   (forces the other assignment; always the globally best choice).
pub(super) fn select_branch_var<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    node: &Node<C>,
    instance: &Arc<HuboInstance<C, V>>,
    constraint_handler: &ConstraintHandler,
    spin_core: Option<Vec<usize>>,
    config: &Config<Lb>,
    incumbent: Option<C>,
) -> BranchChoice {
    // Merge the fresh GE core (from the current node's bound computation) with
    // the inherited hint stored in lb_warm_start.  The fresh core takes
    // priority; the inherited hint is the fallback for non-Cheap bounds.
    let spin_core = spin_core.or_else(|| config.lb.branch_hint(node));

    let sb = &config.strong_branching;

    if !sb.enabled || node.fixed.num_free() < sb.min_free_vars {
        return match node.next_branch_var(instance, constraint_handler, spin_core) {
            Some(v) => BranchChoice::On(v),
            None => BranchChoice::Leaf,
        };
    }

    log::trace!(
        "Selecting branching variable among top {} candidates",
        sb.max_candidates
    );

    let candidates =
        node.branching_candidates(instance, constraint_handler, spin_core, sb.max_candidates);

    if candidates.is_empty() {
        return BranchChoice::Leaf;
    }

    let parent_lb = node.lb.to_f64();
    let tol = config.optimality_tol;

    let mut best_var = candidates[0]; // fallback: top heuristic candidate
    let mut best_score = f64::NEG_INFINITY;

    for var in candidates {
        let mut feasible = [true, true];
        let mut child_lbs = [parent_lb, parent_lb];

        for (i, high) in [false, true].into_iter().enumerate() {
            let mut child = node.child(instance, var, high);

            if !propagate_constraints_only(instance, constraint_handler, &mut child) {
                feasible[i] = false;
                continue;
            }

            let lb = instance
                .round_lower_bound_to_objective_grid(cheap_lb(instance, &mut child))
                .to_f64()
                .max(parent_lb);
            child_lbs[i] = lb;

            // Count prunable children the same as infeasible ones — either way
            // branching on this variable closes that side of the tree.
            if incumbent.is_some_and(|inc| lb + tol >= inc.to_f64()) {
                feasible[i] = false;
            }
        }

        match feasible {
            [false, false] => return BranchChoice::Infeasible,
            // One side is closed — forcing the other assignment is the best
            // possible choice; stop evaluating further candidates.
            [false, true] | [true, false] => return BranchChoice::On(var),
            [true, true] => {
                // Product rule: maximise (Δlb₀ + ε) × (Δlb₁ + ε).
                // The epsilon keeps variables with zero improvement comparable
                // and avoids degenerate 0 × anything = 0 ties.
                let imp0 = (child_lbs[0] - parent_lb).max(0.0);
                let imp1 = (child_lbs[1] - parent_lb).max(0.0);
                let score = (imp0 + 1e-6) * (imp1 + 1e-6);
                if score > best_score {
                    best_score = score;
                    best_var = var;
                }
            }
        }
    }

    BranchChoice::On(best_var)
}
