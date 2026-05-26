use std::{collections::HashMap, sync::Arc};

use crate::{
    domain::VarDomain,
    fixes::{FixError, Fixes},
};

use super::*;

fn coeff_to_json_value<C: Coeff>(c: C) -> serde_json::Value {
    let f = c.to_f64();
    if f.is_finite() && f.fract() == 0.0 {
        serde_json::Value::Number(serde_json::Number::from(f as i64))
    } else {
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}

fn coeff_to_json_value_opt<C: Coeff>(c: Option<C>) -> serde_json::Value {
    match c {
        Some(v) => coeff_to_json_value(v),
        None => serde_json::Value::Null,
    }
}

// Note: Config<Lb> has no bound on Lb here to avoid circular dependency
// (bounds/mod.rs imports from solver::bnb). The LowerBound bound is applied
// at usage sites in bounds/mod.rs and solve.rs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Optimal,
    TimeLimit,
    NodeLimit,
    Cutoff,
    Interrupted,
}

pub struct SolveResult<C: Coeff> {
    pub status: Status,
    pub objective: Option<C>,
    pub best_bound: C,
    pub solution: Option<BitSolution>,
    pub solving_time: f64,
    pub tts: Option<f64>,
    pub n_nodes: u64,
    pub pruned_nodes: u64,
    pub unexplored_nodes: u64,
}

impl<C: Coeff> SolveResult<C> {
    pub fn write_solution_file(&self, path: impl AsRef<Path>, var_type: VarType) -> io::Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let sol_arr = self.solution.as_ref().map(|s| s.to_json_array(var_type));
            let obj_val = coeff_to_json_value_opt(self.objective);
            let bound_val = coeff_to_json_value(self.best_bound);
            let obj = serde_json::json!({
                "status": format!("{:?}", self.status),
                "objective": obj_val,
                "best_bound": bound_val,
                "time_s": self.solving_time,
                "tts_s": self.tts,
                "nodes_explored": self.n_nodes,
                "nodes_pruned": self.pruned_nodes,
                "nodes_unexplored": self.unexplored_nodes,
                "solution": sol_arr,
            });
            return std::fs::write(path, serde_json::to_string_pretty(&obj).unwrap());
        }

        let mut f = std::fs::File::create(path)?;
        writeln!(f, "# HUES solution file")?;
        writeln!(f, "STATUS {:?}", self.status)?;
        if let Some(obj) = self.objective {
            writeln!(f, "OBJECTIVE {obj}")?;
        } else {
            writeln!(f, "OBJECTIVE n/a")?;
        }
        writeln!(f, "BEST_BOUND {}", self.best_bound)?;
        writeln!(f, "TIME {:.6}", self.solving_time)?;
        if let Some(tts) = self.tts {
            writeln!(f, "TTS {tts:.6}")?;
        }
        writeln!(f, "NODES {}", self.n_nodes)?;
        if let Some(ref sol) = self.solution {
            writeln!(f, "SOLUTION")?;
            sol.write_to(&mut f, var_type)?;
        }
        Ok(())
    }
}

// ── ProbingConfig ──────────────────────────────────────────────────────────

/// Configuration for probing inside branch-and-bound nodes.
///
/// Probing temporarily fixes each free variable to each value, propagates
/// constraints, and uses the outcomes to derive forced fixings:
/// - If one branch is infeasible → fix the variable to the other value.
/// - If both branches fix another variable to the same value → fix it unconditionally.
/// - If both branches are infeasible → the node itself is infeasible.
///
/// To keep overhead manageable probing is skipped entirely on nodes that
/// already have fewer than `min_free_vars` free variables — those nodes are
/// close to a leaf and rarely benefit.
#[derive(Debug, Clone)]
pub struct ProbingConfig {
    /// Whether probing is enabled.
    pub enabled: bool,
    /// Maximum number of free variables to probe at each BnB node.
    pub max_vars: usize,
    /// Minimum number of free variables required to run probing at a node.
    /// Nodes below this threshold are skipped — they are deep in the tree
    /// and the search cost per node is already low.
    pub min_free_vars: usize,
}

impl Default for ProbingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_vars: 100,
            min_free_vars: 5,
        }
    }
}

// ── StrongBranchingConfig ──────────────────────────────────────────────────

/// Configuration for strong branching inside branch-and-bound nodes.
///
/// Before committing to a branching variable, strong branching evaluates the
/// cheap lower bound for both children of each candidate variable.  The
/// variable that maximises the **product of bound improvements** over the two
/// children is chosen — the standard reliability-branching score that balances
/// the quality of each branch.
///
/// Nodes with fewer than `min_free_vars` free variables skip strong branching
/// (overhead outweighs benefit near leaves).
#[derive(Debug, Clone)]
pub struct StrongBranchingConfig {
    pub enabled: bool,
    /// Maximum number of candidate variables to evaluate.
    /// Candidates are pre-ranked by heuristic score (parity core, constraints).
    pub max_candidates: usize,
    /// Skip strong branching when fewer than this many variables are free.
    pub min_free_vars: usize,
}

impl Default for StrongBranchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_candidates: 10,
            min_free_vars: 10,
        }
    }
}

// ── Config ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Config<Lb> {
    pub time_limit: Option<f64>,
    pub node_limit: Option<u64>,
    pub cutoff: Option<f64>,
    pub progress_every_nodes: Option<u64>,
    pub stats_csv: Option<String>,
    pub instance_name: Option<String>,
    pub solution_file: Option<String>,
    /// Lower-bounding oracle to use at every BnB node.
    pub lb: Lb,
    /// Run heuristic warm-starts (SA + Tabu) before exact search.
    pub warm_start_heuristics: bool,
    /// Per-heuristic warm-start time budget in seconds.
    pub warm_start_heuristic_time_limit: Option<f64>,
    /// Number of parallel worker threads.  1 = serial (default).
    pub n_threads: usize,
    /// Kernelization configuration applied to individual BnB nodes.
    pub node_kernelization: KernelizationConfig,
    /// Probing configuration applied at each BnB node before branching.
    pub probing: ProbingConfig,
    /// Strong branching configuration applied at each BnB node when selecting the branch variable.
    pub strong_branching: StrongBranchingConfig,
    /// Absolute gap tolerance: a node is pruned when `lb + optimality_tol >= ub`.
    /// Set to 0.0 for exact optimality.
    pub optimality_tol: f64,
    /// Run root-level kernelization before branch-and-bound.
    pub kernelization: bool,
    /// RNG seed for warm-start heuristics.  The same seed produces the same
    /// initial incumbent on every run, making the solver fully deterministic.
    pub seed: u64,
    /// Minimum improvement in the global lower bound (as a percentage of the
    /// gap `|ub − lb|`) required to emit a "bound" log row.
    /// 0.0 = log every improvement; 1.0 = log only when gap closes by ≥ 1 %.
    pub bound_log_min_improvement_pct: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ConstraintHandler {
    pub(crate) constraints: Vec<Box<dyn Constraint>>,
}

impl ConstraintHandler {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }
}

/// Status of a term in the objective function.
#[derive(Clone, Debug, PartialEq)]
pub struct PartiallyAssignedTerm<C: Coeff> {
    /// The variables in the term that are still free (not yet fixed by the partial assignment).
    pub free_variables: Vec<usize>,
    /// Effective coefficient after applying fixed variables (sign flips for SPIN, zeroing for BIN).
    pub coeff: C,
}

impl<C: Coeff> PartiallyAssignedTerm<C> {
    pub fn new(term: &Term<C>) -> Self {
        Self {
            free_variables: term.indices.clone(),
            coeff: term.coeff,
        }
    }

    /// Update the term status by fixing a variable to a value.
    /// Returns `Some(value)` if the term is resolved (fully assigned or zeroed), or `None` if it is still active.
    pub fn set_variable<V: VarDomain>(&mut self, variable: usize, high: bool) -> Option<C> {
        V::update_partial_term(&mut self.coeff, &mut self.free_variables, variable, high)
    }
}

// ── Node ───────────────────────────────────────────────────────────────────

/// A self-contained BnB tree node.
///
/// Each node **owns** a snapshot of the partial variable assignment so that
/// it can be stored on the explicit frontier, cloned, and handed off to
/// an independent worker thread without any mutable back-references.
///
/// `assigned[i]` is set iff variable `i` has been fixed at this node.
/// `values[i]` encodes the "high" flag: BIN `true` → 1, SPIN `true` → +1.
#[derive(Clone)]
pub struct Node<C: Coeff> {
    /// Fixes stored in two FixedbitSets for memory-efficient cloning and thread-safe ownership.
    pub fixed: Fixes,
    /// Lower bound on the objective over this node's sub-tree.
    pub lb: C,
    /// The offset coming from fully fixed terms
    pub offset: C,
    /// A list of unfixed terms
    pub term_status: Vec<Option<PartiallyAssignedTerm<C>>>,
    /// Merge-detection index: maps each active term's free-variable set to its index
    /// in `term_status`.  `None` on freshly created / frontier nodes; built lazily on
    /// the first `set_variable` call so that cloning a node never pays this cost.
    pub(crate) term_by_free_vars: Option<HashMap<Vec<usize>, usize>>,
    /// Local (node-specific) hard constraints/cuts that apply only to this subtree.
    pub(crate) local_constraints: ConstraintHandler,
    /// Opaque warm-start state for the current lower-bounding method.
    ///
    /// Each `LowerBound` impl stores its own concrete type here (e.g.
    /// `SubgradLambda`, `LpBasis`, or a GE core hint for `Cheap`).  The Arc
    /// makes cloning a node — and thus propagating warm-start data to child
    /// nodes — a single reference-count increment.  The child replaces this
    /// field with a new Arc after computing its own bound.
    pub(crate) lb_warm_start: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl<C: Coeff> PartialEq for Node<C> {
    fn eq(&self, other: &Self) -> bool {
        self.lb == other.lb
    }
}
impl<C: Coeff> Eq for Node<C> {}
impl<C: Coeff> PartialOrd for Node<C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<C: Coeff> Ord for Node<C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed so that lower lb = higher priority in BinaryHeap (max-heap).
        other
            .lb
            .partial_cmp(&self.lb)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<C: Coeff> Node<C> {
    /// Build the free-variable-set → term-index map from a `term_status` slice.
    /// Only active (Some) terms are indexed; their `free_variables` vecs are the keys.
    pub(crate) fn build_free_var_map(
        term_status: &[Option<PartiallyAssignedTerm<C>>],
    ) -> HashMap<Vec<usize>, usize> {
        term_status
            .iter()
            .enumerate()
            .filter_map(|(idx, ts)| ts.as_ref().map(|t| (t.free_variables.clone(), idx)))
            .collect()
    }

    /// Create the root node, applying any pre-fixed variables to `term_status`.
    pub fn root<V: VarDomain>(instance: Arc<HuboInstance<C, V>>, lb: C) -> Self {
        let term_status: Vec<Option<PartiallyAssignedTerm<C>>> = instance
            .terms
            .iter()
            .map(|term| Some(PartiallyAssignedTerm::new(term)))
            .collect();

        Self {
            fixed: Fixes::new(instance.n_vars()),
            lb,
            offset: C::zero(),
            term_status,
            term_by_free_vars: None,
            local_constraints: ConstraintHandler::new(),
            lb_warm_start: None,
        }
    }

    /// Fix a variable in-place at this node.
    /// Updates the term status accordingly and appends the fixing to `self.path`.
    pub fn set_variable<V: VarDomain>(
        &mut self,
        instance: &Arc<HuboInstance<C, V>>,
        var: usize,
        high: bool,
    ) -> Result<(), FixError> {
        if self.fixed.assigned.contains(var) {
            log::warn!("set_variable called on already assigned variable");
            return Err(FixError::ConflictingFixes { index: var });
        }

        self.fixed.set(var, high)?;

        // Build the merge-detection map lazily on first use so that cloning a node
        // (child creation) never pays this cost — only nodes that actually fix a
        // variable need the map.
        if self.term_by_free_vars.is_none() {
            self.term_by_free_vars = Some(Self::build_free_var_map(&self.term_status));
        }
        let map = self.term_by_free_vars.as_mut().unwrap();

        // update term status
        for &term_idx in &instance.var_terms[var] {
            if self.term_status[term_idx].is_none() {
                continue;
            }

            // Remove the term's current map entry before its free_variables change.
            let old_key = self.term_status[term_idx]
                .as_ref()
                .unwrap()
                .free_variables
                .clone();
            map.remove(&old_key);

            if let Some(term_value) = self.term_status[term_idx]
                .as_mut()
                .unwrap()
                .set_variable::<V>(var, high)
            {
                // Term fully resolved — offset accumulates, nothing to re-index.
                self.offset += term_value;
                self.term_status[term_idx] = None;
            } else {
                // Term still active with new (smaller) free-variable set.
                // O(log n) lookup: is there already another term with the same free vars?
                let new_key = self.term_status[term_idx]
                    .as_ref()
                    .unwrap()
                    .free_variables
                    .clone();

                if let Some(&merge_idx) = map.get(&new_key) {
                    // Merge: absorb this term's coefficient into the surviving term.
                    let coeff = self.term_status[term_idx].as_ref().unwrap().coeff;
                    self.term_status[term_idx] = None;
                    let survivor = self.term_status[merge_idx].as_mut().unwrap();
                    survivor.coeff += coeff;
                    // If the surviving term's coefficient cancelled to zero, resolve it too.
                    if survivor.coeff == C::zero() {
                        map.remove(&new_key);
                        self.term_status[merge_idx] = None;
                    }
                } else {
                    map.insert(new_key, term_idx);
                }
            }
        }

        Ok(())
    }

    /// Propagate all constraints in `self.constraint_handler` until convergence or infeasibility.
    /// Returns `false` if an infeasibility is detected, `true` otherwise.
    pub(crate) fn propagate_constraints<V: VarDomain>(
        &mut self,
        instance: &Arc<HuboInstance<C, V>>,
        constraint_handler: &ConstraintHandler,
    ) -> bool {
        propagate_constraints_only(instance, constraint_handler, self)
    }

    pub fn set_lower_bound(&mut self, new_lb: C) {
        if new_lb > self.lb {
            self.lb = new_lb;
        }
    }

    /// Return a child node where `var` is fixed to `high` with the given `lb`.
    pub fn child<V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        var: usize,
        high: bool,
    ) -> Self {
        let mut child_node = Self {
            fixed: self.fixed.clone(),
            lb: self.lb,
            offset: self.offset,
            term_status: self.term_status.clone(),
            term_by_free_vars: None,
            local_constraints: self.local_constraints.clone(),
            lb_warm_start: self.lb_warm_start.clone(),
        };

        // set variable and update term status/parity knapsack
        child_node
            .set_variable(instance, var, high)
            .map_err(|err| {
                log::error!("Error fixing variable in child node: {:?}", err);
                err
            })
            .ok();

        // log number of constraints/cuts in the child node after fixing the variable, for debugging
        log::debug!(
            "child node created by fixing var {} to {}, lb = {}",
            var,
            high,
            self.lb,
        );

        child_node
    }

    /// Index of the first unassigned variable, or `None` at a leaf.
    #[inline]
    pub fn next_unassigned(&self) -> Option<usize> {
        self.fixed.assigned.zeroes().next()
    }

    /// Return up to `max_k` candidate branching variables, ranked by descending
    /// heuristic score.  Priority order (additive):
    ///
    /// 1. **Active-term count** (+1 per live term containing the variable) —
    ///    favours high-connectivity variables whose fixings affect the most terms.
    /// 2. **Parity core** (+1 000 per term in the unsat core, spin only) —
    ///    strongly prefers variables involved in the cheapest known contradiction.
    /// 3. **Constraint activity** (+1 per active cut/constraint containing the var).
    ///
    /// Variables with score 0 (isolated variables with no active terms and no
    /// constraint involvement) are appended as a last resort in index order.
    ///
    /// Used by strong branching to build the candidate set before evaluating
    /// both children of each variable with the cheap lower bound.
    pub(crate) fn branching_candidates<V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        constraint_handler: &ConstraintHandler,
        spin_core: Option<Vec<usize>>,
        max_k: usize,
    ) -> Vec<usize> {
        let mut scores = vec![0u64; instance.n_vars()];

        // Active-term count: +1 per term that is still live at this node and
        // contains the variable.  Branching on a variable with high connectivity
        // tends to produce large LB improvements in both children.
        for v in self.fixed.iter_free() {
            let active_count = instance.var_terms[v]
                .iter()
                .filter(|&&ti| self.term_status[ti].is_some())
                .count();
            scores[v] += active_count as u64;
        }

        if V::VAR_TYPE == VarType::Spin {
            // spin_core is pre-resolved by the caller (select_branch_var merges
            // the current-computation core with the inherited lb_warm_start hint).
            if let Some(core_terms) = spin_core {
                for ti in core_terms {
                    for &v in &instance.terms[ti].indices {
                        if !self.fixed.assigned.contains(v) {
                            scores[v] += 1_000;
                        }
                    }
                }
            }
        }

        constraint_handler.accumulate_branch_scores(self, &mut scores);
        self.local_constraints
            .accumulate_branch_scores(self, &mut scores);

        // Collect scored free variables; sort descending by score.
        let mut scored: Vec<(usize, u64)> = scores
            .iter()
            .enumerate()
            .filter(|&(v, &s)| !self.fixed.assigned.contains(v) && s > 0)
            .map(|(v, &s)| (v, s))
            .collect();
        scored.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));

        let mut result: Vec<usize> = scored.into_iter().map(|(v, _)| v).collect();

        // Fill remaining slots with unscored free variables (first-unassigned order).
        if result.len() < max_k {
            for v in self.fixed.iter_free() {
                if result.len() >= max_k {
                    break;
                }
                if scores[v] == 0 {
                    result.push(v);
                }
            }
        }

        result.truncate(max_k);
        result
    }

    /// Pick the single best branching variable by heuristic score.
    /// Delegates to [`branching_candidates`] with `max_k = 1`.
    #[inline]
    pub(crate) fn next_branch_var<V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        constraint_handler: &ConstraintHandler,
        spin_core: Option<Vec<usize>>,
    ) -> Option<usize> {
        self.branching_candidates(instance, constraint_handler, spin_core, 1)
            .into_iter()
            .next()
    }

    /// Materialise the fully-assigned node as a [`BitSolution`].
    ///
    /// Panics in debug mode if any variable is still free.
    pub fn to_bitsolution<V: VarDomain>(&self, instance: &Arc<HuboInstance<C, V>>) -> BitSolution {
        debug_assert!(
            (0..instance.n_vars()).all(|i| self.fixed.assigned.contains(i)),
            "to_bitsolution called on incomplete assignment"
        );
        BitSolution {
            values: self.fixed.values.clone(),
        }
    }

    /// Convert the partial assignment to `Vec<Option<C>>` for Lasserre routines.
    pub fn to_option_vec<V: VarDomain>(&self, instance: &HuboInstance<C, V>) -> Vec<Option<C>> {
        (0..instance.n_vars())
            .map(|i| {
                if self.fixed.assigned.contains(i) {
                    let high = self.fixed.values.contains(i);
                    Some(V::high_to_coeff::<C>(high))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_fixes<V: VarDomain>(&self, instance: &Arc<HuboInstance<C, V>>) -> Vec<(usize, C)> {
        (0..instance.n_vars())
            .filter(|&i| self.fixed.assigned.contains(i))
            .map(|i| {
                let high = self.fixed.values.contains(i);
                (i, V::high_to_coeff::<C>(high))
            })
            .collect()
    }

    // pub fn lift_solution_to_source(
    //     &self,
    //     bitsol: &BitSolution,
    //     instance: Arc<HuboInstance<C, V>>,
    // ) -> Result<BitSolution, KernelizationError> {
    //     let reduced = bitsol.to_vec(instance.var_type);
    //     let lifted = self.kernel_trace.lift_solution(&reduced)?;
    //     Ok(BitSolution::from_vec(&lifted))
    // }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{instance::HuboInstance, term::Term};

    use super::{Node, PartiallyAssignedTerm};

    fn make_bin_instance(
        n_vars: usize,
        terms: Vec<Term<i64>>,
    ) -> Arc<HuboInstance<i64, crate::domain::Bin>> {
        Arc::new(HuboInstance::new(n_vars, 0, terms))
    }

    fn term(indices: Vec<usize>, coeff: i64) -> Term<i64> {
        Term { indices, coeff }
    }

    fn active(node: &Node<i64>, idx: usize) -> Option<(Vec<usize>, i64)> {
        node.term_status[idx]
            .as_ref()
            .map(|ts: &PartiallyAssignedTerm<i64>| (ts.free_variables.clone(), ts.coeff))
    }

    // Two terms share free vars only after a variable is fixed in each;
    // this is the primary case the merge exists to handle.
    #[test]
    fn merge_two_terms_with_same_free_vars_after_fix() {
        // 2·x0·x1·x2 + 3·x0·x1·x3  →  fix x2=1, fix x3=1  →  merge on {x0,x1}, coeff 5
        let instance = make_bin_instance(4, vec![term(vec![0, 1, 2], 2), term(vec![0, 1, 3], 3)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 2, true).unwrap();
        node.set_variable(&instance, 3, true).unwrap();

        let t0 = active(&node, 0);
        let t1 = active(&node, 1);
        assert!(
            (t0 == Some((vec![0, 1], 5)) && t1.is_none())
                || (t1 == Some((vec![0, 1], 5)) && t0.is_none()),
            "expected merge into coeff 5 on {{0,1}}, got t0={t0:?} t1={t1:?}"
        );
    }

    // Singleton free-variable merge: both terms reduce to a single shared variable.
    #[test]
    fn singleton_merge() {
        // 2·x0·x1 + 3·x0·x2  →  fix x1=1, fix x2=1  →  merge on {x0}, coeff 5
        let instance = make_bin_instance(3, vec![term(vec![0, 1], 2), term(vec![0, 2], 3)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 1, true).unwrap();
        node.set_variable(&instance, 2, true).unwrap();

        let t0 = active(&node, 0);
        let t1 = active(&node, 1);
        assert!(
            (t0 == Some((vec![0], 5)) && t1.is_none())
                || (t1 == Some((vec![0], 5)) && t0.is_none()),
            "expected merge into coeff 5 on {{0}}, got t0={t0:?} t1={t1:?}"
        );
    }

    // Terms that reduce to different free-variable sets must not be merged.
    #[test]
    fn no_merge_when_free_vars_differ() {
        // 2·x0·x1·x2 + 3·x0·x2·x3  →  fix x2=1  →  term0={x0,x1}, term1={x0,x3}, no merge
        let instance = make_bin_instance(4, vec![term(vec![0, 1, 2], 2), term(vec![0, 2, 3], 3)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 2, true).unwrap();

        assert_eq!(active(&node, 0), Some((vec![0, 1], 2)));
        assert_eq!(active(&node, 1), Some((vec![0, 3], 3)));
    }

    // A candidate that shares all free variables but has additional ones must not be merged.
    #[test]
    fn no_merge_when_candidate_has_extra_vars() {
        // 2·x0·x1·x2 + 3·x0·x1·x2·x3  →  fix x2=1  →  term0={x0,x1}, term1={x0,x1,x3}
        let instance =
            make_bin_instance(4, vec![term(vec![0, 1, 2], 2), term(vec![0, 1, 2, 3], 3)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 2, true).unwrap();

        assert_eq!(active(&node, 0), Some((vec![0, 1], 2)));
        assert_eq!(active(&node, 1), Some((vec![0, 1, 3], 3)));
    }

    // Merging into a term that does NOT contain the fixed variable.
    // When {x0,x1,x2} loses x2 its free-var set becomes {x0,x1}, which is
    // already present as a shorter term.  The map lookup must find that
    // shorter term even though it never appears in var_terms[x2].
    #[test]
    fn merge_into_term_not_containing_fixed_var() {
        // 2·x0·x1·x2 + 3·x0·x1  →  fix x2=1  →  merge on {x0,x1}, coeff 5
        let instance = make_bin_instance(3, vec![term(vec![0, 1, 2], 2), term(vec![0, 1], 3)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 2, true).unwrap();

        let t0 = active(&node, 0);
        let t1 = active(&node, 1);
        assert!(
            (t0 == Some((vec![0, 1], 5)) && t1.is_none())
                || (t1 == Some((vec![0, 1], 5)) && t0.is_none()),
            "expected merge into coeff 5 on {{0,1}}, got t0={t0:?} t1={t1:?}"
        );
    }

    // Fixing a variable to 0 (binary) zeroes the coefficient and resolves the term.
    #[test]
    fn fix_to_zero_resolves_term() {
        let instance = make_bin_instance(2, vec![term(vec![0, 1], 5)]);
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 1, false).unwrap();

        assert!(node.term_status[0].is_none());
        assert_eq!(node.offset, 0);
    }

    // Three terms where only two should merge; the third has a different free-var set.
    #[test]
    fn merge_only_matching_pair_among_three_terms() {
        // 2·x0·x1·x2 + 3·x0·x1·x3 + 4·x0·x2·x3
        // fix x2=1: term0→{x0,x1}, term2→{x0,x3}
        // fix x3=1: term1→{x0,x1} — merges with term0; term2→{x0} (resolved from {x0,x3})
        let instance = make_bin_instance(
            4,
            vec![
                term(vec![0, 1, 2], 2),
                term(vec![0, 1, 3], 3),
                term(vec![0, 2, 3], 4),
            ],
        );
        let mut node = Node::root(instance.clone(), 0i64);

        node.set_variable(&instance, 2, true).unwrap();
        node.set_variable(&instance, 3, true).unwrap();

        let t0 = active(&node, 0);
        let t1 = active(&node, 1);
        let t2 = active(&node, 2);

        // term0 and term1 must have merged into coeff 5 on {x0,x1}
        assert!(
            (t0 == Some((vec![0, 1], 5)) && t1.is_none())
                || (t1 == Some((vec![0, 1], 5)) && t0.is_none()),
            "expected merge of term0+term1, got t0={t0:?} t1={t1:?}"
        );
        // term2 becomes {x0} with coeff 4 (x2=1 and x3=1, still one free var)
        assert_eq!(t2, Some((vec![0], 4)));
    }
}
