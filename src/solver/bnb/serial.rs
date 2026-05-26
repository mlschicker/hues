use crate::solver::bnb::util::log_table_row;

use super::*;

pub(super) struct SearchState<'a, C: Coeff, V: CoverCutDomain, Lb: LowerBound> {
    pub(crate) instance: Arc<HuboInstance<C, V>>,
    pub(crate) constraint_handler: ConstraintHandler,
    pub(crate) config: &'a Config<Lb>,
    pub(crate) start: Instant,
    pub(crate) root_lb: C,
    /// Min lb over all open nodes; refreshed lazily on log events.
    pub(crate) cached_global_lb: C,
    pub(crate) incumbent_obj: Option<C>,
    pub(crate) incumbent_sol: Option<BitSolution>,
    pub(crate) tts: Option<f64>,
    pub(crate) explored_nodes: u64,
    pub(crate) pruned_nodes: u64,
    pub(crate) leaf_nodes: u64,
    /// Number of subproblems closed by Gray-code enumeration.
    pub(crate) enum_gc_nodes: u64,
    pub(crate) last_log_nodes: u64,
    /// Tracks the last global-lb value that triggered a "bound" log row.
    pub(crate) last_logged_lb: C,
    /// Optional channel for receiving solution hints from another solver.
    pub(crate) solution_receiver: &'a mpsc::Receiver<Vec<C>>,
    pub(crate) stop_status: Option<Status>,
}

impl<'a, C: Coeff, V: CoverCutDomain, Lb: LowerBound> SearchState<'a, C, V, Lb> {
    fn global_lb(&self, frontier: &BinaryHeap<Node<C>>, active_lb: Option<C>) -> C {
        let frontier_min = frontier.peek().map(|n| n.lb);
        match (frontier_min, active_lb) {
            (Some(f), Some(a)) => {
                if a < f {
                    a
                } else {
                    f
                }
            }
            (Some(f), None) => f,
            (None, Some(a)) => a,
            (None, None) => self.root_lb,
        }
    }

    /// Returns `true` if `lb` is within `optimality_tol` of `ub`, meaning the
    /// node can be pruned without affecting optimality up to the tolerance.
    #[inline]
    fn prune_by_lb(&self, lb: C, ub: C) -> bool {
        lb.to_f64() + self.config.optimality_tol >= ub.to_f64()
    }

    /// Check stopping criteria and return the triggered status, if any.
    /// Checks the following in order: interrupt signal, time limit, node limit, incumbent cutoff.
    pub(crate) fn should_stop(&self) -> Option<Status> {
        if crate::interrupt::is_interrupted() {
            return Some(Status::Interrupted);
        }
        if let Some(limit) = self.config.time_limit
            && self.start.elapsed().as_secs_f64() >= limit
        {
            return Some(Status::TimeLimit);
        }
        if let Some(limit) = self.config.node_limit
            && self.explored_nodes >= limit
        {
            return Some(Status::NodeLimit);
        }
        if let (Some(cutoff), Some(obj)) = (self.config.cutoff, self.incumbent_obj)
            && obj.to_f64() <= cutoff
        {
            return Some(Status::Cutoff);
        }
        None
    }

    /// Update the incumbent solution and log the event if improved.
    pub(crate) fn update_incumbent(
        &mut self,
        objective: C,
        bitsol: BitSolution,
        event: &str,
        frontier_size: usize,
    ) {
        // Check for improvement
        let improved = match self.incumbent_obj {
            Some(best) => objective < best,
            None => true,
        };

        // Update necessary information
        if improved {
            self.incumbent_obj = Some(objective);
            self.incumbent_sol = Some(bitsol);
            if self.tts.is_none() {
                self.tts = Some(self.start.elapsed().as_secs_f64());
            }
            log_table_row(self, event, self.cached_global_lb, frontier_size);
        }
    }

    /// Drain all pending solutions from `solution_receiver` and update the
    /// incumbent if any of them improve on the current best.
    pub(crate) fn maybe_import_live_solution(
        &mut self,
        frontier: &BinaryHeap<Node<C>>,
        active_lb: Option<C>,
    ) {
        let received: Vec<Vec<C>> = {
            let mut buf = Vec::new();
            while let Ok(src) = self.solution_receiver.try_recv() {
                buf.push(src);
            }
            buf
        };

        for source_vals in received {
            let bitsol = BitSolution::from_vec(&source_vals);
            let objective = bitsol.evaluate(&self.instance);

            self.refresh_global_lb(frontier, active_lb);
            let fsz = frontier.len();
            self.update_incumbent(objective, bitsol, "injected", fsz);
        }
    }

    /// Refresh `cached_global_lb` by scanning the frontier. This is called lazily on log events to avoid
    /// the overhead of maintaining a global lb on every node push/pop, but can be useful to get a more accurate gap estimate in the logs.
    pub(crate) fn refresh_global_lb(
        &mut self,
        frontier: &BinaryHeap<Node<C>>,
        active_lb: Option<C>,
    ) {
        let glb = self.global_lb(frontier, active_lb);
        if glb > self.cached_global_lb {
            self.cached_global_lb = glb;
        }
    }

    pub(crate) fn maybe_log_progress(
        &mut self,
        frontier: &BinaryHeap<Node<C>>,
        active_lb: Option<C>,
    ) {
        let by_nodes = self
            .config
            .progress_every_nodes
            .is_some_and(|s| s > 0 && self.explored_nodes.saturating_sub(self.last_log_nodes) >= s);
        if !by_nodes {
            return;
        }

        self.refresh_global_lb(frontier, active_lb);
        log_table_row(self, "progress", self.cached_global_lb, frontier.len());
        self.last_log_nodes = self.explored_nodes;
    }

    pub(crate) fn maybe_log_bound_improvement(
        &mut self,
        frontier: &BinaryHeap<Node<C>>,
        active_lb: Option<C>,
    ) {
        self.refresh_global_lb(frontier, active_lb);
        let new_lb = self.cached_global_lb.to_f64();
        let old_lb = self.last_logged_lb.to_f64();
        if new_lb <= old_lb {
            return;
        }
        let threshold = self.config.bound_log_min_improvement_pct;
        let passes = if threshold <= 0.0 {
            true
        } else if let Some(ub) = self.incumbent_obj {
            let ub = ub.to_f64();
            let gap_pct_of = |lb: f64| {
                let denom = ub.abs().max(lb.abs()).max(1e-12);
                ((ub - lb).max(0.0) / denom) * 100.0
            };
            gap_pct_of(old_lb) - gap_pct_of(new_lb) >= threshold
        } else {
            false
        };
        if passes {
            self.last_logged_lb = self.cached_global_lb;
            log_table_row(self, "bound", self.cached_global_lb, frontier.len());
        }
    }

    /// Solve a node with ≤ [`enumerate::GRAY_CODE_THRESHOLD`] free variables
    /// inline via Gray-code enumeration.
    /// The node is treated as fully explored after this returns; no children
    /// are pushed back to the frontier.  Always emits a table row tagged
    /// `enum-gc` so the caller can see enumeration progress.
    pub(crate) fn solve_small_node(&mut self, node: Node<C>, frontier: &BinaryHeap<Node<C>>) {
        let problem = enumerate::LocalProblem::build(self.instance.as_ref(), &node);
        let var_type = self.instance.var_type();
        // let n_free = problem.n_free();
        // let (best_obj, best_pattern) = if n_free >= 8 {
        //     let nthreads = rayon::current_num_threads();
        //     let k = (nthreads.ilog2() as usize).max(1).min(n_free - 1);
        //     enumerate::gray_code_solve_parallel(&problem, var_type, k)
        // } else {
        //     enumerate::gray_code_solve(&problem, var_type)
        // };

        let (best_obj, best_pattern) = enumerate::gray_code_solve(&problem, var_type);

        self.leaf_nodes += 1;
        self.enum_gc_nodes += 1;

        let mut values = node.fixed.values.clone();
        for (i, &g) in problem.local_to_global.iter().enumerate() {
            values.set(g, (best_pattern >> i) & 1 == 1);
        }
        let bitsol = BitSolution { values };
        let objective = bitsol.evaluate(self.instance.as_ref());
        self.refresh_global_lb(frontier, None);
        self.update_incumbent(objective, bitsol, "enum-gc", frontier.len());
        let _ = best_obj;
    }

    // ── Iterative search loop ──────────────────────────────────────────────

    /// Run the search.  Returns `true` when optimal (frontier exhausted),
    /// `false` when stopped early (remaining nodes stay in `frontier`).
    pub(crate) fn run_loop(&mut self, frontier: &mut BinaryHeap<Node<C>>) -> bool {
        while let Some(mut node) = frontier.pop() {
            // Check stopping criteria
            if let Some(status) = self.should_stop() {
                self.stop_status = Some(status);
                frontier.push(node);
                return false;
            }

            self.explored_nodes += 1;
            self.maybe_import_live_solution(frontier, Some(node.lb));

            if self
                .incumbent_obj
                .is_some_and(|best| self.prune_by_lb(node.lb, best))
            {
                self.pruned_nodes += 1;
                continue;
            }

            self.maybe_log_progress(frontier, Some(node.lb));
            self.maybe_log_bound_improvement(frontier, Some(node.lb));

            kernelize_search_node(&self.instance, &mut node, self.config, self.incumbent_obj);

            if !node.propagate_constraints(&self.instance, &self.constraint_handler) {
                self.pruned_nodes += 1;
                continue;
            }

            log::trace!(
                "BnB node: explored={} best={} lb={}",
                self.explored_nodes,
                self.incumbent_obj
                    .map_or("n/a".to_string(), |v| v.to_f64().to_string()),
                node.lb.to_f64()
            );

            if let Some(best) = self.incumbent_obj
                && !propagate_by_incumbent(
                    &self.instance,
                    &mut node,
                    best,
                    &self.constraint_handler,
                )
            {
                self.pruned_nodes += 1;
                continue;
            }

            if !probe_node(
                &self.instance,
                &mut node,
                &self.constraint_handler,
                self.incumbent_obj,
                &self.config.probing,
            ) {
                self.pruned_nodes += 1;
                continue;
            }

            let (new_lb, spin_core) = compute_node_lb_with_core(
                &mut node,
                self.config,
                self.incumbent_obj,
                &self.instance,
            );

            node.set_lower_bound(new_lb);

            if self
                .incumbent_obj
                .is_some_and(|best| self.prune_by_lb(node.lb, best))
            {
                self.pruned_nodes += 1;
                continue;
            }


            let n_free = node.fixed.num_free();
            if n_free > 0 && n_free <= enumerate::GRAY_CODE_THRESHOLD {
                self.solve_small_node(node, frontier);
                self.maybe_log_bound_improvement(frontier, None);
                continue;
            }

            match select_branch_var(
                &node,
                &self.instance,
                &self.constraint_handler,
                spin_core,
                self.config,
                self.incumbent_obj,
            ) {
                BranchChoice::On(var) => {
                    for high in [true, false] {
                        let mut child = node.child(&self.instance, var, high);

                        if !child.propagate_constraints(&self.instance, &self.constraint_handler) {
                            self.pruned_nodes += 1;
                            continue;
                        }

                        let new_lb = compute_node_lb(
                            &mut child,
                            self.config,
                            self.incumbent_obj,
                            &self.instance,
                        );
                        child.set_lower_bound(new_lb);
                        if self
                            .incumbent_obj
                            .is_some_and(|best| self.prune_by_lb(child.lb, best))
                        {
                            self.pruned_nodes += 1;
                            continue;
                        }

                        child.term_by_free_vars = None;
                        frontier.push(child);
                    }
                }
                BranchChoice::Leaf => {
                    self.leaf_nodes += 1;
                    let bitsol = node.to_bitsolution(&self.instance);
                    let objective = bitsol.evaluate(&self.instance);
                    self.refresh_global_lb(frontier, None);
                    self.update_incumbent(objective, bitsol, "incumbent", frontier.len());
                }
                BranchChoice::Infeasible => {
                    self.pruned_nodes += 1;
                }
            }
        }
        true
    }
}
