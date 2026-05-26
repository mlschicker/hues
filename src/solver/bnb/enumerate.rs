//! Enumerative subproblem solvers used inside BnB nodes that have shrunk
//! to a tractable number of free variables.
//!
//! Two paths are provided:
//! - [`gray_code_solve`] for ≤ [`GRAY_CODE_THRESHOLD`] free vars: walks all
//!   `2^n` assignments via a binary reflected Gray code, maintaining the
//!   running objective by deltas.  Unconditional `O(2^n · avg_deg)` work.
//! - [`gray_code_solve_parallel`]: same walk split across `2^k` rayon tasks,
//!   each fixing the first `k` variables to a distinct prefix and walking the
//!   remaining `n − k` variables.  No sub-problem is materialised.

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;

use super::types::Node;

/// Subproblems with at most this many free variables are handled inline by
/// the Gray-code enumerator.  Larger subproblems stay in the BnB frontier.
pub(super) const GRAY_CODE_THRESHOLD: usize = 25;

/// Compact representation of a small subproblem: only active terms,
/// reindexed over free variables.
pub(super) struct LocalProblem<C: Coeff> {
    /// `local_to_global[i]` is the original variable index for local var `i`.
    pub(super) local_to_global: Vec<usize>,
    /// Active terms as `(coeff, free-var indices in local space)`.
    pub(super) terms: Vec<(C, Vec<u8>)>,
    /// `var_terms[v]` lists term indices touching local var `v`.
    pub(super) var_terms: Vec<Vec<u32>>,
    /// Constant contribution from `instance.offset + node.offset`.
    pub(super) offset: C,
}

impl<C: Coeff> LocalProblem<C> {
    pub(super) fn build<V: VarDomain>(instance: &HuboInstance<C, V>, node: &Node<C>) -> Self {
        let n_vars = instance.n_vars();
        let local_to_global: Vec<usize> = node.fixed.assigned.zeroes().collect();
        debug_assert!(
            local_to_global.len() <= 32,
            "LocalProblem assumes at most 32 free variables"
        );

        let mut global_to_local = vec![u8::MAX; n_vars];
        for (i, &g) in local_to_global.iter().enumerate() {
            global_to_local[g] = i as u8;
        }

        let mut terms: Vec<(C, Vec<u8>)> = Vec::new();
        for ts in node.term_status.iter().flatten() {
            let local_vars: Vec<u8> = ts
                .free_variables
                .iter()
                .map(|&v| {
                    debug_assert_ne!(global_to_local[v], u8::MAX);
                    global_to_local[v]
                })
                .collect();
            terms.push((ts.coeff, local_vars));
        }

        let n_local = local_to_global.len();
        let mut var_terms: Vec<Vec<u32>> = vec![Vec::new(); n_local];
        for (ti, term) in terms.iter().enumerate() {
            for &v in &term.1 {
                var_terms[v as usize].push(ti as u32);
            }
        }

        Self {
            local_to_global,
            terms,
            var_terms,
            offset: instance.offset + node.offset,
        }
    }

    pub(super) fn n_free(&self) -> usize {
        self.local_to_global.len()
    }
}

/// Walk all `2^n` assignments of the free variables via a binary reflected
/// Gray code, tracking the running objective by deltas.  Returns
/// `(best_obj, best_pattern)` where `best_pattern` bit `i` is set iff local
/// var `i` holds its high value (BIN = 1, SPIN = +1).
pub(super) fn gray_code_solve<C: Coeff>(problem: &LocalProblem<C>, var_type: VarType) -> (C, u32) {
    let n = problem.n_free();
    debug_assert!(n > 0 && n <= 32);
    let n_terms = problem.terms.len();
    let total = 1u32 << n;
    let mut state: u32 = 0;

    match var_type {
        VarType::Bin => {
            // Initial state: all free vars at 0; no term activated.
            let mut counts = vec![0u8; n_terms];
            let mut cur_obj = problem.offset;
            let mut best_obj = cur_obj;
            let mut best_state = state;

            for k in 1..total {
                let v = k.trailing_zeros() as usize;
                let was_one = (state >> v) & 1 == 1;
                if !was_one {
                    for &ti in &problem.var_terms[v] {
                        let ti_us = ti as usize;
                        let term = &problem.terms[ti_us];
                        counts[ti_us] += 1;
                        if counts[ti_us] as usize == term.1.len() {
                            cur_obj += term.0;
                        }
                    }
                } else {
                    for &ti in &problem.var_terms[v] {
                        let ti_us = ti as usize;
                        let term = &problem.terms[ti_us];
                        if counts[ti_us] as usize == term.1.len() {
                            cur_obj -= term.0;
                        }
                        counts[ti_us] -= 1;
                    }
                }
                state ^= 1u32 << v;

                if cur_obj < best_obj {
                    best_obj = cur_obj;
                    best_state = state;
                }
            }

            (best_obj, best_state)
        }
        VarType::Spin => {
            // Initial state: all free vars at -1.  Term contribution is
            // coeff · (-1)^|free_vars|.
            let mut term_contrib: Vec<C> = problem
                .terms
                .iter()
                .map(|(c, vars)| if vars.len() % 2 == 0 { *c } else { -*c })
                .collect();
            let mut cur_obj = problem.offset;
            for &c in &term_contrib {
                cur_obj += c;
            }
            let mut best_obj = cur_obj;
            let mut best_state = state;

            for k in 1..total {
                let v = k.trailing_zeros() as usize;
                for &ti in &problem.var_terms[v] {
                    let ti_us = ti as usize;
                    let old = term_contrib[ti_us];
                    let new = -old;
                    term_contrib[ti_us] = new;
                    cur_obj = cur_obj + new - old;
                }
                state ^= 1u32 << v;

                if cur_obj < best_obj {
                    best_obj = cur_obj;
                    best_state = state;
                }
            }

            (best_obj, best_state)
        }
    }
}

/// Parallel Gray-code solver: enumerate all `2^n` assignments by fixing the
/// first `k` variables to each prefix in `0..2^k` and walking the remaining
/// `n − k` variables via Gray code in independent rayon tasks.
///
/// No sub-problem is materialised.  Each task initialises per-term state from
/// the prefix and then walks local variables `k..n-1` directly on `problem`.
pub(super) fn gray_code_solve_parallel<C: Coeff>(
    problem: &LocalProblem<C>,
    var_type: VarType,
    k: usize,
) -> (C, u32) {
    debug_assert!(
        k > 0 && k < problem.n_free(),
        "k={k} must be in 1..n_free={}",
        problem.n_free()
    );
    debug_assert!(problem.n_free() <= 32);

    use rayon::prelude::*;

    let n = problem.n_free();
    let n_tail = n - k;
    let total_tail = 1u32 << n_tail;

    (0u32..1u32 << k)
        .into_par_iter()
        .map(|prefix| match var_type {
            VarType::Bin => {
                let n_terms = problem.terms.len();
                // Per-term high-count initialised from the prefix.
                // For a term with no suffix vars, its activation is absorbed
                // into init_obj directly; counts[ti] is left at 0 and never touched.
                let mut counts = vec![0u8; n_terms];
                let mut init_obj = problem.offset;

                for (ti, (coeff, vars)) in problem.terms.iter().enumerate() {
                    let prefix_high = vars
                        .iter()
                        .filter(|&&v| (v as usize) < k && (prefix >> v) & 1 == 1)
                        .count();
                    let has_suffix = vars.iter().any(|&v| (v as usize) >= k);
                    if !has_suffix {
                        // Fully resolved by the prefix.
                        if prefix_high == vars.len() {
                            init_obj += *coeff;
                        }
                    } else {
                        counts[ti] = prefix_high as u8;
                    }
                }

                let mut cur_obj = init_obj;
                let mut best_obj = cur_obj;
                let mut suffix_state: u32 = 0;
                let mut best_suffix: u32 = 0;

                for step in 1..total_tail {
                    let tail_bit = step.trailing_zeros() as usize;
                    let v = k + tail_bit;
                    let was_one = (suffix_state >> tail_bit) & 1 == 1;
                    if !was_one {
                        for &ti in &problem.var_terms[v] {
                            let ti_us = ti as usize;
                            counts[ti_us] += 1;
                            if counts[ti_us] as usize == problem.terms[ti_us].1.len() {
                                cur_obj += problem.terms[ti_us].0;
                            }
                        }
                    } else {
                        for &ti in &problem.var_terms[v] {
                            let ti_us = ti as usize;
                            if counts[ti_us] as usize == problem.terms[ti_us].1.len() {
                                cur_obj -= problem.terms[ti_us].0;
                            }
                            counts[ti_us] -= 1;
                        }
                    }
                    suffix_state ^= 1u32 << tail_bit;
                    if cur_obj < best_obj {
                        best_obj = cur_obj;
                        best_suffix = suffix_state;
                    }
                }

                (best_obj, prefix | (best_suffix << k))
            }
            VarType::Spin => {
                // Initial term contributions: all free vars start at -1.
                // Contribution of term ti = coeff * (-1)^arity.
                // Each prefix var set to +1 negates all terms it appears in.
                // Net: coeff * (-1)^(arity + prefix_high_count).
                let mut term_contrib: Vec<C> = problem
                    .terms
                    .iter()
                    .map(|(coeff, vars)| {
                        let prefix_high = vars
                            .iter()
                            .filter(|&&v| (v as usize) < k && (prefix >> v) & 1 == 1)
                            .count();
                        if (vars.len() + prefix_high) % 2 == 0 {
                            *coeff
                        } else {
                            -*coeff
                        }
                    })
                    .collect();

                let mut cur_obj = problem.offset;
                for &c in &term_contrib {
                    cur_obj += c;
                }
                let mut best_obj = cur_obj;
                let mut suffix_state: u32 = 0;
                let mut best_suffix: u32 = 0;

                for step in 1..total_tail {
                    let tail_bit = step.trailing_zeros() as usize;
                    let v = k + tail_bit;
                    for &ti in &problem.var_terms[v] {
                        let ti_us = ti as usize;
                        let old = term_contrib[ti_us];
                        let new = -old;
                        term_contrib[ti_us] = new;
                        cur_obj = cur_obj + new - old;
                    }
                    suffix_state ^= 1u32 << tail_bit;
                    if cur_obj < best_obj {
                        best_obj = cur_obj;
                        best_suffix = suffix_state;
                    }
                }

                (best_obj, prefix | (best_suffix << k))
            }
        })
        .reduce_with(|(obj_a, pat_a), (obj_b, pat_b)| {
            if obj_b < obj_a { (obj_b, pat_b) } else { (obj_a, pat_a) }
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::domain::{Bin, Spin};
    use crate::instance::HuboInstance;
    use crate::solution::BitSolution;
    use crate::term::Term;

    fn make_bin(n_vars: usize, terms: Vec<(Vec<usize>, i64)>) -> Arc<HuboInstance<i64, Bin>> {
        let terms = terms
            .into_iter()
            .map(|(idx, c)| Term {
                indices: idx,
                coeff: c,
            })
            .collect();
        Arc::new(HuboInstance::new(n_vars, 0i64, terms))
    }

    fn make_spin(n_vars: usize, terms: Vec<(Vec<usize>, i64)>) -> Arc<HuboInstance<i64, Spin>> {
        let terms = terms
            .into_iter()
            .map(|(idx, c)| Term {
                indices: idx,
                coeff: c,
            })
            .collect();
        Arc::new(HuboInstance::new(n_vars, 0i64, terms))
    }

    fn brute_best_bin(instance: &HuboInstance<i64, Bin>) -> i64 {
        let n = instance.n_vars();
        let mut best = i64::MAX;
        for pat in 0u32..(1u32 << n) {
            let mut sol = BitSolution::new(n);
            for i in 0..n {
                sol.values.set(i, (pat >> i) & 1 == 1);
            }
            let obj = sol.evaluate(instance);
            if obj < best {
                best = obj;
            }
        }
        best
    }

    fn brute_best_spin(instance: &HuboInstance<i64, Spin>) -> i64 {
        let n = instance.n_vars();
        let mut best = i64::MAX;
        for pat in 0u32..(1u32 << n) {
            let mut sol = BitSolution::new(n);
            for i in 0..n {
                sol.values.set(i, (pat >> i) & 1 == 1);
            }
            let obj = sol.evaluate(instance);
            if obj < best {
                best = obj;
            }
        }
        best
    }

    fn pattern_to_solution(n_vars: usize, local_to_global: &[usize], pattern: u32) -> BitSolution {
        let mut sol = BitSolution::new(n_vars);
        for (i, &g) in local_to_global.iter().enumerate() {
            sol.values.set(g, (pattern >> i) & 1 == 1);
        }
        sol
    }

    #[test]
    fn gray_code_bin_matches_brute_force() {
        let inst = make_bin(
            4,
            vec![
                (vec![0, 1], 2),
                (vec![1, 2], -3),
                (vec![0, 2, 3], 4),
                (vec![3], 5),
            ],
        );
        let node = Node::root(inst.clone(), 0i64);
        let problem = LocalProblem::build(inst.as_ref(), &node);
        let (obj, pattern) = gray_code_solve(&problem, VarType::Bin);
        let sol = pattern_to_solution(inst.n_vars(), &problem.local_to_global, pattern);
        assert_eq!(sol.evaluate(inst.as_ref()), obj);
        assert_eq!(obj, brute_best_bin(inst.as_ref()));
    }

    #[test]
    fn gray_code_spin_matches_brute_force() {
        let inst = make_spin(
            4,
            vec![
                (vec![0, 1], 2),
                (vec![1, 2], -3),
                (vec![0, 2, 3], 4),
                (vec![3], 5),
            ],
        );
        let node = Node::root(inst.clone(), 0i64);
        let problem = LocalProblem::build(inst.as_ref(), &node);
        let (obj, pattern) = gray_code_solve(&problem, VarType::Spin);
        let sol = pattern_to_solution(inst.n_vars(), &problem.local_to_global, pattern);
        assert_eq!(sol.evaluate(inst.as_ref()), obj);
        assert_eq!(obj, brute_best_spin(inst.as_ref()));
    }

    #[test]
    fn parallel_gray_code_bin_matches_serial() {
        let inst = make_bin(
            6,
            vec![
                (vec![0, 1], 2),
                (vec![1, 2], -3),
                (vec![0, 2, 3], 4),
                (vec![3, 4], -1),
                (vec![4, 5], 5),
                (vec![0, 5], -2),
            ],
        );
        let node = Node::root(inst.clone(), 0i64);
        let problem = LocalProblem::build(inst.as_ref(), &node);
        let (serial_obj, _) = gray_code_solve(&problem, VarType::Bin);
        for k in 1..problem.n_free() {
            let (par_obj, par_pat) = gray_code_solve_parallel(&problem, VarType::Bin, k);
            let sol = pattern_to_solution(inst.n_vars(), &problem.local_to_global, par_pat);
            assert_eq!(
                sol.evaluate(inst.as_ref()),
                par_obj,
                "k={k}: evaluated obj mismatch"
            );
            assert_eq!(
                par_obj, serial_obj,
                "k={k}: parallel obj differs from serial"
            );
        }
    }

    #[test]
    fn parallel_gray_code_spin_matches_serial() {
        let inst = make_spin(
            6,
            vec![
                (vec![0, 1], 2),
                (vec![1, 2], -3),
                (vec![0, 2, 3], 4),
                (vec![3, 4], -1),
                (vec![4, 5], 5),
                (vec![0, 5], -2),
            ],
        );
        let node = Node::root(inst.clone(), 0i64);
        let problem = LocalProblem::build(inst.as_ref(), &node);
        let (serial_obj, _) = gray_code_solve(&problem, VarType::Spin);
        for k in 1..problem.n_free() {
            let (par_obj, par_pat) = gray_code_solve_parallel(&problem, VarType::Spin, k);
            let sol = pattern_to_solution(inst.n_vars(), &problem.local_to_global, par_pat);
            assert_eq!(
                sol.evaluate(inst.as_ref()),
                par_obj,
                "k={k}: evaluated obj mismatch"
            );
            assert_eq!(
                par_obj, serial_obj,
                "k={k}: parallel obj differs from serial"
            );
        }
    }
}
