use crate::fixes::Fixes;
use crate::{Bin, Spin};

use super::*;

// ── Bound propagation / cutting planes ────────────────────────────────────

impl ConstraintHandler {
    pub(crate) fn add_constraint(&mut self, constraint: Box<dyn Constraint>) {
        let key = constraint.key();
        if !self
            .constraints
            .iter()
            .any(|existing| existing.key() == key)
        {
            self.constraints.push(constraint);
        }
    }

    pub(crate) fn add_parity_cut(&mut self, free_vars: Vec<usize>, odd_required: bool) {
        self.add_constraint(Box::new(ParityConstraint {
            free_vars,
            odd_required,
        }));
    }

    pub(crate) fn add_cover_cut(&mut self, cut: CoverConstraint) {
        self.add_constraint(Box::new(cut));
    }

    #[allow(unused)]
    pub(crate) fn add_lex_order_cut(&mut self, vars: Vec<usize>) {
        if vars.len() < 2 {
            return;
        }
        self.add_constraint(Box::new(LexOrderConstraint { vars }));
    }

    pub(crate) fn add_lex_comparison_cut(&mut self, pairs: Vec<(usize, usize)>) {
        if pairs.is_empty() {
            return;
        }
        self.add_constraint(Box::new(LexComparisonConstraint { pairs }));
    }

    /// Remove irrelevant constraints and detect contradictions for fully
    /// resolved constraints.
    pub(crate) fn cleanup_and_check(&mut self, fixed: &Fixes) -> bool {
        let mut kept: Vec<Box<dyn Constraint>> = Vec::with_capacity(self.constraints.len());
        for c in &self.constraints {
            match c.cleanup_and_check(fixed) {
                ConstraintCleanup::Keep => {
                    kept.push(c.clone());
                }
                ConstraintCleanup::Drop => {}
                ConstraintCleanup::Infeasible => return false,
            }
        }
        self.constraints = kept;
        true
    }

    pub(crate) fn accumulate_branch_scores<C: Coeff>(&self, node: &Node<C>, scores: &mut [u64]) {
        for c in &self.constraints {
            c.accumulate_branch_scores(&node.fixed, scores);
        }
    }

    pub(crate) fn propagate_round<C: Coeff>(
        &self,
        node: &Node<C>,
        fixed_this_round: &mut Fixes,
    ) -> ConstraintPropagation {
        // initialize flags
        let mut changed = false;
        let mut progress = true;

        while progress {
            // reset flag
            progress = false;

            // iterate over constraints and propagate
            for c in &self.constraints {
                match c.propagate(&node.fixed, fixed_this_round) {
                    ConstraintPropagation::Infeasible => return ConstraintPropagation::Infeasible,
                    ConstraintPropagation::Fixed => {
                        log::trace!(
                            "constraint propagation deduced new variable fixings: {:?} from constraint {:?}",
                            fixed_this_round,
                            c.key()
                        );
                        changed = true;
                        progress = true;
                    }
                    ConstraintPropagation::NoChange => {}
                }
            }
        }

        // return whether any variable was fixed in this round
        if changed {
            ConstraintPropagation::Fixed
        } else {
            ConstraintPropagation::NoChange
        }
    }
}

/// Add one lex-comparison constraint per generator permutation: enforce `a ≤_lex p(a)`.
///
/// For each generator permutation `p`, this adds the constraint that the assignment `a`
/// is lexicographically ≤ its image `p(a)`. This is always a valid symmetry break for
/// any permutation group — no assumption is made about the group structure (cyclic,
/// symmetric, etc.). It correctly handles products of disjoint transpositions like the
/// reversal symmetry of LABS, where individual transpositions are NOT symmetries on their own.
///
/// Returns the number of inserted constraints.
pub(super) fn add_lex_comparison_from_permutations(
    constraint_handler: &mut ConstraintHandler,
    permutations: &[Vec<usize>],
) -> usize {
    let mut added = 0usize;

    for perm in permutations {
        let n = perm.len();

        // Compute the inverse permutation p⁻¹.
        let mut inv = vec![0usize; n];
        for (i, &pi) in perm.iter().enumerate() {
            inv[pi] = i;
        }

        // Pairs (k, p⁻¹[k]) in position order, skipping fixed points.
        let pairs: Vec<(usize, usize)> = (0..n)
            .filter(|&k| inv[k] != k)
            .map(|k| (k, inv[k]))
            .collect();

        if pairs.is_empty() {
            continue;
        }

        constraint_handler.add_lex_comparison_cut(pairs);
        added += 1;
    }
    added
}

/// Propagate already learned hard constraints until a fixed point.
///
/// Returns `false` if the constraint set proves infeasibility under the
/// current partial assignment, otherwise `true`.
pub(super) fn propagate_constraints_only<C: Coeff, V: VarDomain>(
    instance: &Arc<HuboInstance<C, V>>,
    constraint_handler: &ConstraintHandler,
    node: &mut Node<C>,
) -> bool {
    let n_vars = instance.n_vars();
    let mut any_fixed_in_round = true;
    let mut fixed_this_round = Fixes::new(n_vars);

    while any_fixed_in_round {
        any_fixed_in_round = false;
        fixed_this_round.clear();

        // propagate constraints (global + local) to a fixed point
        let mut progress = true;
        while progress {
            progress = false;

            match constraint_handler.propagate_round(node, &mut fixed_this_round) {
                ConstraintPropagation::Infeasible => return false,
                ConstraintPropagation::Fixed => {
                    progress = true;
                }
                ConstraintPropagation::NoChange => {}
            }

            match node
                .local_constraints
                .propagate_round(node, &mut fixed_this_round)
            {
                ConstraintPropagation::Infeasible => return false,
                ConstraintPropagation::Fixed => {
                    progress = true;
                }
                ConstraintPropagation::NoChange => {}
            }
        }

        // apply the variable fixings from this round to the node's assignment and term status,
        // and update flags for the next iteration.
        for (var, high) in fixed_this_round.iter_fixed() {
            // set the variable in the node
            if node.set_variable(instance, var, high).is_err() {
                return false;
            }

            // update flags
            any_fixed_in_round = true;
        }

        // cleanup constraints and check for contradictions before the next round of propagation,
        // which may be able to deduce more with the new fixings.
        if !node.local_constraints.cleanup_and_check(&node.fixed) {
            return false;
        }
    }

    // node.lb = lower_bound(instance, node);
    true
}

/// Generate incumbent-based knapsack cover cuts for a binary node
///
/// Split the objective into
///   S1 = unresolved terms with positive coefficient and no zero fixed yet
///        (the knapsack items)
///   S2 = everything else
///
/// From `f(x) ≤ incumbent` and lower bound `f2_lb` on S2:
///
///   ∑_{S∈S1} c_S · T_S  ≤  incumbent − f2_lb   (=: RHS)
///
/// Greedy cover: sort S1 by coefficient descending, accumulate until
/// sum > RHS, then minimise (remove items that are redundant).
///
/// Returns up to one cover cut per call (the tightest single greedy cover).
pub trait CoverCutDomain: VarDomain {
    fn gen_cover_constraints<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        node: &Node<C>,
        incumbent: C,
    ) -> Vec<CoverConstraint>;

    /// Splittable per-term lower bound for incumbent analysis.
    /// For BIN this is the full decomposable bound; for SPIN this excludes the non-decomposable parity bonus.
    fn incumbent_analysis_base_lb<C: Coeff>(instance: &HuboInstance<C, Self>, node: &Node<C>) -> C;
}

impl CoverCutDomain for Bin {
    fn gen_cover_constraints<C: Coeff>(
        _instance: &HuboInstance<C, Self>,
        node: &Node<C>,
        incumbent: C,
    ) -> Vec<CoverConstraint> {
        let mut f2_lb = C::zero();
        let mut s1: Vec<(Vec<usize>, C)> = Vec::new();

        for term_state in &node.term_status {
            let Some(term_status) = term_state else {
                continue;
            };

            if term_status.free_variables.is_empty() {
                // Fully assigned
                f2_lb += term_status.coeff;
                continue;
            }

            // Partially assigned
            if term_status.coeff == C::zero() {
                continue;
            }
            if term_status.coeff > C::zero() {
                s1.push((term_status.free_variables.clone(), term_status.coeff));
            } else {
                f2_lb += term_status.coeff;
            }
        }

        if s1.is_empty() {
            return Vec::new();
        }

        let rhs = incumbent - f2_lb;

        s1.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cover_sum = C::zero();
        let mut cover: Vec<usize> = Vec::new();
        for (idx, &(_, c)) in s1.iter().enumerate() {
            cover_sum += c;
            cover.push(idx);
            if cover_sum > rhs {
                break;
            }
        }
        if cover_sum <= rhs {
            return Vec::new();
        }

        let mut i = 0;
        while i < cover.len() {
            let c = s1[cover[i]].1;
            if cover_sum - c > rhs {
                cover_sum -= c;
                cover.remove(i);
            } else {
                i += 1;
            }
        }

        let items: Vec<Vec<usize>> = cover.iter().map(|&idx| s1[idx].0.clone()).collect();
        let max_active = items.len().saturating_sub(1);

        log::debug!(
            "cover cut: {} items, max_active={}, cover_sum={}, rhs={}",
            items.len(),
            max_active,
            cover_sum.to_f64(),
            rhs.to_f64()
        );

        vec![CoverConstraint { items, max_active }]
    }

    fn incumbent_analysis_base_lb<C: Coeff>(instance: &HuboInstance<C, Self>, node: &Node<C>) -> C {
        lower_bound(instance, node)
    }
}

impl CoverCutDomain for Spin {
    fn gen_cover_constraints<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        node: &Node<C>,
        incumbent: C,
    ) -> Vec<CoverConstraint> {
        // For spin terms, base LB uses term_lb = -|c|.
        // Any parity violating the preferred one incurs a penalty 2|c|.
        // We build a conservative binary surrogate z_i <= y_i where
        // z_i = 1 iff all free vars are +1. This implies parity-even, so z_i <= y_i
        // holds for terms whose preferred parity is odd.
        let lb_base = lower_bound_spin_base(instance, node);

        let mut s1: Vec<(Vec<usize>, C)> = Vec::new();
        for term_state in &node.term_status {
            let Some(term_status) = term_state else {
                continue;
            };

            if term_status.free_variables.is_empty() {
                continue;
            }

            let coeff = term_status.coeff;
            if coeff == C::zero() {
                continue;
            }

            // Preferred parity among free vars that attains term_lb = -|c|.
            let odd_required = coeff > C::zero();
            if !odd_required {
                continue;
            }

            let penalty = (coeff + coeff).abs();
            if penalty <= C::zero() {
                continue;
            }

            s1.push((term_status.free_variables.clone(), penalty));
        }

        if s1.is_empty() {
            return Vec::new();
        }

        let rhs = incumbent - lb_base;

        s1.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cover_sum = C::zero();
        let mut cover: Vec<usize> = Vec::new();
        for (idx, &(_, c)) in s1.iter().enumerate() {
            cover_sum += c;
            cover.push(idx);
            if cover_sum > rhs {
                break;
            }
        }
        if cover_sum <= rhs {
            return Vec::new();
        }

        let mut i = 0;
        while i < cover.len() {
            let c = s1[cover[i]].1;
            if cover_sum - c > rhs {
                cover_sum -= c;
                cover.remove(i);
            } else {
                i += 1;
            }
        }

        let items: Vec<Vec<usize>> = cover.iter().map(|&idx| s1[idx].0.clone()).collect();
        let max_active = items.len().saturating_sub(1);

        log::debug!(
            "spin cover cut: {} items, max_active={}, cover_sum={}, rhs={}",
            items.len(),
            max_active,
            cover_sum.to_f64(),
            rhs.to_f64()
        );

        vec![CoverConstraint { items, max_active }]
    }

    fn incumbent_analysis_base_lb<C: Coeff>(instance: &HuboInstance<C, Self>, node: &Node<C>) -> C {
        lower_bound_spin_base(instance, node)
    }
}

/// Tighten a node's partial assignment by using the incumbent upper bound as a
/// per-term constraint.
/// returns false if infeasibility is detected, true otherwise. May fix variables and add parity cuts.
pub(super) fn propagate_by_incumbent<C: Coeff, V: CoverCutDomain>(
    instance: &Arc<HuboInstance<C, V>>,
    node: &mut Node<C>,
    incumbent: C,
    constraint_handler: &ConstraintHandler,
) -> bool {
    // let initial_cover_cuts = V::gen_cover_constraints(instance, node, incumbent);

    // // TODO: Initial cover cuts should be added to constraint_handler if it's available
    // // for now they're collected but not applied

    // for cut in initial_cover_cuts {
    //     node.local_constraints.add_cover_cut(cut);
    // }

    // Splittable per-term bound: decomposable so lb_without analysis is valid.
    // For BIN this is the full cheap bound; for SPIN this excludes the non-decomposable parity bonus.
    let mut lb_base = V::incumbent_analysis_base_lb(instance, node);
    // Pruning bound: seed with the better of the stored node bound (which may come from an
    // expensive non-decomposable oracle) and the current termwise bound rounded to the objective grid.
    let mut prune_lb = node
        .lb
        .max_of(instance.round_lower_bound_to_objective_grid(lb_base));

    let mut fixed_this_round = Fixes::new(instance.n_vars());

    loop {
        if prune_lb >= incumbent {
            return false;
        }

        fixed_this_round.clear();
        let mut term_states = node.term_status.clone();
        let mut any_fixed = false;

        let mut learned_parity_cuts: Vec<ParityConstraint> = Vec::new();
        let mut learned_cover_cuts: Vec<CoverConstraint> = Vec::new();

        for term_idx in 0..instance.terms.len() {
            // Skip terms that are fully assigned or already None
            let (free_variables, coeff) = {
                if let Some(term_status) = &term_states[term_idx] {
                    if term_status.free_variables.is_empty() {
                        continue;
                    }
                    (term_status.free_variables.clone(), term_status.coeff)
                } else {
                    continue;
                }
            };
            if coeff == C::zero() {
                continue;
            }

            if V::VAR_TYPE == VarType::Bin {
                if let Some(feasible) = propagate_term_by_incumbent_bin::<C, V>(
                    &free_variables,
                    incumbent,
                    coeff,
                    lb_base,
                    instance,
                    &mut fixed_this_round,
                    &mut term_states,
                    &mut any_fixed,
                    &mut learned_cover_cuts,
                ) {
                    return feasible;
                }
            } else {
                if free_variables.is_empty() {
                    continue;
                }

                if let Some(feasible) = propagate_term_by_incumbent_spin::<C, V>(
                    &free_variables,
                    incumbent,
                    coeff,
                    lb_base,
                    instance,
                    &mut fixed_this_round,
                    &mut term_states,
                    &mut any_fixed,
                    &mut learned_parity_cuts,
                ) {
                    return feasible;
                }
            }
        }

        // Add any newly learned cuts as local constraints for this subtree.
        if !learned_parity_cuts.is_empty() {
            for cut in learned_parity_cuts {
                log::trace!(
                    "learned parity cut from incumbent-based analysis: free_vars={:?}, odd_required={}",
                    cut.free_vars,
                    cut.odd_required
                );
                node.local_constraints
                    .add_parity_cut(cut.free_vars, cut.odd_required);
            }
        }

        if !learned_cover_cuts.is_empty() {
            for cut in learned_cover_cuts {
                node.local_constraints.add_cover_cut(cut);
            }
        }

        // Propagate global + local cuts together until no further progress can be made.
        let mut progress = true;
        while progress {
            progress = false;

            match constraint_handler.propagate_round(node, &mut fixed_this_round) {
                ConstraintPropagation::Infeasible => return false,
                ConstraintPropagation::Fixed => {
                    any_fixed = true;
                    progress = true;
                }
                ConstraintPropagation::NoChange => {}
            }

            match node
                .local_constraints
                .propagate_round(node, &mut fixed_this_round)
            {
                ConstraintPropagation::Infeasible => return false,
                ConstraintPropagation::Fixed => {
                    any_fixed = true;
                    progress = true;
                }
                ConstraintPropagation::NoChange => {}
            }
        }

        // Apply the variable fixings from this round to the node's assignment and term status.

        for (var, high) in fixed_this_round.iter_fixed() {
            if node.set_variable(instance, var, high).is_err() {
                return false;
            }
        }

        // Cleanup local constraints and check for contradictions before the next round.
        if !node.local_constraints.cleanup_and_check(&node.fixed) {
            return false;
        }

        // If no new variable fixings were deduced in this round, then we cannot make further progress by propagating the incumbent-based cuts, so we can stop here.

        if !any_fixed {
            break;
        }

        // Recompute the lower bound with the new partial assignment before the next round of propagation, which may be able to deduce more with the tighter bound.
        // For spin, also recompute the base lb for the per-term analysis.

        lb_base = V::incumbent_analysis_base_lb(instance, node);
        prune_lb = prune_lb.max_of(instance.round_lower_bound_to_objective_grid(lb_base));
    }

    node.lb = prune_lb;
    true
}

/// Analyze a single term with free variables against the incumbent to deduce variable fixings or cuts for the binary case.
fn propagate_term_by_incumbent_bin<C: Coeff, V: VarDomain>(
    free_variables: &[usize],
    incumbent: C,
    term_coeff: C,
    lb: C,
    instance: &HuboInstance<C, V>,
    fixed_this_round: &mut Fixes,
    term_states: &mut [Option<PartiallyAssignedTerm<C>>],
    any_fixed: &mut bool,
    learned_cover_cuts: &mut Vec<CoverConstraint>,
) -> Option<bool> {
    let t_one = term_coeff;
    let t_zero = C::zero();

    // get the lower bound without the term's contribution, to compare against the incumbent for the one and zero cases
    let term_lb = if t_one <= t_zero { t_one } else { t_zero };
    let lb_without = lb - term_lb;

    let one_pruned = lb_without + t_one >= incumbent;
    let zero_pruned = lb_without + t_zero >= incumbent;

    if one_pruned && zero_pruned {
        return Some(false);
    }

    if one_pruned {
        // Can only set variable to 0 if one is left
        // otherwise, we get that at least one of the remaining variables must be 0
        if free_variables.len() == 1 {
            let xi = free_variables[0];
            if !apply_fix(fixed_this_round, xi, false) {
                return Some(false);
            }
            for &ti in &instance.var_terms[xi] {
                if let Some(term_status) = &mut term_states[ti] {
                    term_status.set_variable::<V>(xi, false);
                }
            }
            *any_fixed = true;
        }
        // println!("Hello, I am a test.");
        // Doesn't this also introduce a kind of parity cut, i.e., at least one of the free variables must be 0?
        // For now we just rely on the constraint handler to deduce that, but maybe we could add an explicit parity cut here for efficiency?
        learned_cover_cuts.push(CoverConstraint {
            items: vec![free_variables.to_vec()],
            max_active: free_variables.len() - 1,
        });
    } else if zero_pruned {
        // If the term cannot become 0, then all free variables must be set to 1
        let vars_to_fix = free_variables;
        for xi in vars_to_fix {
            if !apply_fix(fixed_this_round, *xi, true) {
                return Some(false);
            }
            for &ti in &instance.var_terms[*xi] {
                if let Some(term_status) = &mut term_states[ti] {
                    term_status.set_variable::<V>(*xi, true);
                }
            }
        }
        *any_fixed = true;
    }
    None
}


/// Analyze a single term with free variables against the incumbent to deduce variable fixings or cuts for the spin case.
fn propagate_term_by_incumbent_spin<C: Coeff, V: VarDomain>(
    free_variables: &[usize],
    incumbent: C,
    term_coeff: C,
    lb_base: C,
    instance: &HuboInstance<C, V>,
    fixed_this_round: &mut Fixes,
    term_states: &mut [Option<PartiallyAssignedTerm<C>>],
    any_fixed: &mut bool,
    learned_parity_cuts: &mut Vec<ParityConstraint>,
) -> Option<bool> {
    let (t_even, t_odd) = (term_coeff, -term_coeff);

    // get the lower bound without the term's contribution, to compare against the incumbent for the even and odd parity cases
    let term_lb = -term_coeff.abs();
    let lb_without = lb_base - term_lb;

    let even_pruned = lb_without + t_even >= incumbent;
    let odd_pruned = lb_without + t_odd >= incumbent;

    log::trace!(
        "incumbent-based analysis of term with free vars {:?}: lb_without={}, t_even={}, t_odd={}, even_pruned={}, odd_pruned={}",
        free_variables,
        lb_without.to_f64(),
        t_even.to_f64(),
        t_odd.to_f64(),
        even_pruned,
        odd_pruned
    );

    // if the incumbent forces both even and odd cases to be infeasible, then the term cannot be satisfied under the current partial assignment
    if even_pruned && odd_pruned {
        return Some(false);
    }

    // If both cases are still feasible, we cannot deduce anything about this term right now.
    if !even_pruned && !odd_pruned {
        return None;
    }

    // Thi leaves the case that exactly one of even/odd is pruned.
    // In that case we can fix the free variables to satisfy the non-pruned parity, or if there are multiple free variables we can at least add a parity cut to exclude the pruned parity.
    let odd_required = even_pruned;

    if free_variables.len() == 1 {
        let high = !odd_required;
        let xi = free_variables[0];
        if !apply_fix(fixed_this_round, xi, high) {
            return Some(false);
        }
        for &ti in &instance.var_terms[xi] {
            if let Some(term_status) = &mut term_states[ti] {
                term_status.set_variable::<V>(xi, high);
            }
        }
        *any_fixed = true;
    } else {
        learned_parity_cuts.push(ParityConstraint {
            free_vars: free_variables.to_vec().clone(),
            odd_required,
        });
    }

    None
}

/// Apply a variable fixing and check for consistency with any previous fixing for the same variable in this round. Returns false if a contradiction is detected, true otherwise.
#[inline]
pub(super) fn apply_fix(fixed_this_round: &mut Fixes, var: usize, high: bool) -> bool {
    match fixed_this_round.get(var) {
        None => fixed_this_round.set(var, high).is_ok(),
        Some(prev) => prev == high,
    }
}
