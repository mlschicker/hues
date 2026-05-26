use super::*;

use crate::fixes::Fixes;
use crate::kernelization::symmetry;
use crate::solver::bnb::util::{
    fmt_coeff, format_gap, log_table_footer, log_table_header, log_table_row,
};

pub(super) struct SearchOutcome<C: Coeff> {
    pub(super) status: Status,
    pub(super) explored: u64,
    pub(super) pruned: u64,
    pub(super) unexplored: u64,
    pub(super) best_bound: C,
    pub(super) incumbent_obj: Option<C>,
    pub(super) incumbent_sol: Option<BitSolution>,
    pub(super) tts: Option<f64>,
}

// ── Warm-start helper ──────────────────────────────────────────────────────

/// Per-heuristic result: `(event_label, objective, elapsed_seconds)` for every
/// heuristic that set a new incumbent.  Used to emit table rows at the root node.
pub(super) type WsImprovements<C> = Vec<(&'static str, C, f64)>;

pub(super) fn run_warm_start<C: Coeff, V: VarDomain, Lb>(
    instance: &HuboInstance<C, V>,
    config: &Config<Lb>,
    initial: Option<(C, BitSolution)>,
    start: Instant,
) -> (Option<(C, BitSolution, f64)>, WsImprovements<C>) {
    let mut best = initial.map(|(obj, sol)| (obj, sol, 0.0));
    let mut improvements: WsImprovements<C> = Vec::new();

    // Skip warm-start if instance has been fully kernelized away
    if instance.n_vars() == 0 {
        return (best, improvements);
    }

    if !config.warm_start_heuristics {
        return (best, improvements);
    }

    let budget = config.warm_start_heuristic_time_limit.unwrap_or_else(|| {
        config
            .time_limit
            .map(|t| (t * 0.05).clamp(0.05, 2.0))
            .unwrap_or(0.5)
    });
    if budget <= 0.0 {
        return (best, improvements);
    }

    // The three heuristics run in cascade: each gets the running best as
    // its starting solution so it refines (rather than rediscovers) the
    // current incumbent.  Diversity comes from each heuristic's
    // additional restarts, which still draw random initial states.
    let n = instance.n_vars();
    let sa_cfg = heuristic::sa::Config {
        common: heuristic::CommonConfig {
            time_limit: None,
            cutoff: config.cutoff,
            seed: Some(config.seed),
            solution_file: None,
        },
        initial_temp: 2.0,
        final_temp: 1e-3,
        cooling_rate: 0.999,
        sweeps_per_temp: Some(n.clamp(1, 64)),
        restarts: 1,
    };
    let sa_init = best.as_ref().map(|(_, s, _)| s.clone());
    let sa = heuristic::sa::solve(instance, &sa_cfg, sa_init.as_ref());
    if best.as_ref().is_none_or(|(b, _, _)| sa.objective < *b) {
        best = Some((sa.objective, sa.solution.clone(), sa.solving_time));
        improvements.push(("ws-sa", sa.objective, start.elapsed().as_secs_f64()));
    }

    let tabu_cfg = heuristic::tabu::Config {
        common: heuristic::CommonConfig {
            time_limit: Some(budget),
            cutoff: config.cutoff,
            seed: Some(config.seed.wrapping_add(1)),
            solution_file: None,
        },
        tenure: None,
        max_iterations: Some((n as u64).saturating_mul(2_000)),
        restarts: 1,
    };
    let tabu_init = best.as_ref().map(|(_, s, _)| s.clone());
    let tabu = heuristic::tabu::solve(instance, &tabu_cfg, tabu_init.as_ref());
    if best.as_ref().is_none_or(|(b, _, _)| tabu.objective < *b) {
        best = Some((tabu.objective, tabu.solution.clone(), tabu.solving_time));
        improvements.push(("ws-tabu", tabu.objective, start.elapsed().as_secs_f64()));
    }

    let mut pt_config = heuristic::parallel_tempering::Config::default();
    pt_config.common.seed = Some(config.seed.wrapping_add(2));
    // Bound PT by the warm-start budget so big instances don't blow past it
    // — PT honors `time_limit` by checking elapsed time after every sweep.
    pt_config.common.time_limit = Some(budget);
    pt_config.common.cutoff = config.cutoff;
    // Seed PT's cold replica with the running best so it refines instead of
    // restarting from random.  PT::solve takes a Vec<C>; convert back from
    // the BitSolution we already have.
    let pt_init = best
        .as_ref()
        .map(|(_, s, _)| s.to_vec::<C>(instance.var_type()));
    let pt = heuristic::parallel_tempering::solve(instance, &pt_config, pt_init);
    if best.as_ref().is_none_or(|(b, _, _)| pt.objective < *b) {
        best = Some((pt.objective, pt.solution, pt.solving_time));
        improvements.push(("ws-pt", pt.objective, start.elapsed().as_secs_f64()));
    }

    log::debug!(
        "heuristic warm-start | {:<6} | obj={:>12} | time={:>8.3}s |",
        "SA",
        sa.objective,
        sa.solving_time
    );
    log::debug!(
        "heuristic warm-start | {:<6} | obj={:>12} | time={:>8.3}s |",
        "Tabu",
        tabu.objective,
        tabu.solving_time
    );
    log::debug!(
        "heuristic warm-start | {:<6} | obj={:>12} | time={:>8.3}s |",
        "PT",
        pt.objective,
        pt.solving_time
    );

    (best, improvements)
}

pub(super) fn kernelize_search_node<C: Coeff, V: VarDomain, Lb: LowerBound>(
    instance: &Arc<HuboInstance<C, V>>,
    node: &mut Node<C>,
    config: &Config<Lb>,
    incumbent: Option<C>,
) {
    let node_free_vars_before = &node.fixed.num_free();
    let node_free_terms_before = node.term_status.iter().filter(|t| t.is_some()).count();

    let kernelizer = Kernelizer::new(config.node_kernelization.clone());

    let report = match kernelizer.kernelize(instance, node, incumbent) {
        Ok(result) => result,
        Err(err) => {
            log::warn!("node kernelization failed, keeping current node: {err}");
            return;
        }
    };

    let node_free_vars_after = &node.fixed.num_free();
    let node_free_terms_after = node.term_status.iter().filter(|t| t.is_some()).count();

    if report.rule_fixed > 0 {
        // Info which kernelization rules were applied at this node, for debugging and future reporting.
        log::debug!(
            "node kernelization: vars {} -> {}, terms {} -> {}, dominance_fixed={}, external={}, fixed_by_rules={} ",
            node_free_vars_before,
            node_free_vars_after,
            node_free_terms_before,
            node_free_terms_after,
            report.dominance_fixed,
            report.externally_fixed,
            report.rule_fixed,
        );
    }

    // Node-level QPBO: apply roof-duality on the degree-≤2 residual when enabled.
    let roof_enabled = match V::VAR_TYPE {
        VarType::Bin => config.node_kernelization.enable_binary_roof_duality,
        VarType::Spin => config.node_kernelization.enable_spin_roof_duality,
    };
    if roof_enabled
        && !apply_roof_dual_fixings(node, instance, binary_roof_duality, spin_roof_duality)
    {
        node.lb = C::max_value();
    }

    // let new_lb = compute_node_lb(node, config, None, instance); // node is already &mut
    // node.set_lower_bound(new_lb);

    // warmstart_lower_bound(node, config, instance);
}

/// Solve the instance using branch-and-bound.
pub fn solve<C: Coeff, V: CoverCutDomain, Lb: LowerBound + Clone>(
    original_instance: &Arc<HuboInstance<C, V>>,
    config: &Config<Lb>,
    initial_sol: Option<Vec<C>>,
    solution_receiver: &mpsc::Receiver<Vec<C>>,
) -> SolveResult<C> {
    // ── Initialization ────────────────────────────────────────────────────────
    let start = Instant::now();

    let term_status: Vec<_> = original_instance
        .terms
        .iter()
        .map(|term| Some(PartiallyAssignedTerm::new(term)))
        .collect();

    let fixed = Fixes::new(original_instance.n_vars());

    let mut root_node_pre = Node {
        fixed,
        lb: C::min_value(),
        offset: C::zero(),
        term_status,
        term_by_free_vars: None,
        local_constraints: ConstraintHandler::new(),
        lb_warm_start: None,
    };

    let mut constraint_handler = ConstraintHandler::new();

    // ── Kernelization ────────────────────────────────────────────────────────
    let (kernel_report, kernel_instance, new_to_old) = if config.kernelization {
        log::info!(
            "kernelizing instance: var_type={:?}, vars={}, terms={}",
            original_instance.var_type(),
            original_instance.n_vars(),
            original_instance.n_terms()
        );

        let start_kernelization = Instant::now();

        let kernelizer = Kernelizer::new(KernelizationConfig::default());
        let report = match kernelizer.kernelize(original_instance, &mut root_node_pre, None) {
            Ok(r) => r,
            Err(err) => {
                log::warn!("kernelization failed, falling back to original instance: {err}");
                KernelizationReport {
                    initial_n_vars: original_instance.n_vars(),
                    final_n_vars: original_instance.n_vars(),
                    ..KernelizationReport::default()
                }
            }
        };

        log::info!("Applying kernelization fixes to instance...");
        let (inst, n2o) = original_instance.apply_fixes(&root_node_pre.fixed);
        log::info!(
            "kernelization complete: vars {} -> {}, terms {} -> {}",
            original_instance.n_vars(),
            inst.n_vars(),
            original_instance.n_terms(),
            inst.n_terms()
        );
        match inst.objective_granularity {
            Some(g) => log::info!(
                "objective granularity: grid spacing = {g}, base = {} \
                 (bounds rounded up to nearest feasible value)",
                inst.objective_grid_base
            ),
            None => log::info!(
                "objective granularity: not available \
                 (float coefficients — bounds are not rounded to an integer grid)"
            ),
        }
        log::info!(
            "kernelization complete: time={:.3}s",
            start_kernelization.elapsed().as_secs_f64()
        );

        (report, inst, n2o)
    } else {
        log::info!("kernelization disabled");
        let n = original_instance.n_vars();
        let (inst, n2o) = original_instance.apply_fixes(&root_node_pre.fixed);
        (
            KernelizationReport {
                initial_n_vars: n,
                final_n_vars: n,
                ..KernelizationReport::default()
            },
            inst,
            n2o,
        )
    };

    // ── Component decomposition ──────────────────────────────────────────────
    // If the kernel's variable-interaction graph is disconnected, split into
    // independent sub-problems and solve each separately.  This can give
    // dramatic speed-ups because exponential search is performed on n/k
    // variables each rather than n variables total.

    let splits = symmetry::split_into_components(&kernel_instance);
    if splits.len() > 1 {
        log::info!(
            "instance decomposes into {} independent components; solving each separately",
            splits.len()
        );
        let result = solve_decomposed(
            original_instance,
            splits,
            &new_to_old,
            &root_node_pre.fixed,
            config,
        );
        if let Some(ref path) = config.solution_file {
            match result.write_solution_file(path, original_instance.var_type()) {
                Ok(()) => log::info!("solution written to {path}"),
                Err(e) => log::error!("failed to write solution file {path}: {e}"),
            }
        }
        if let Some(ref path) = config.stats_csv {
            match append_stats_csv(path, original_instance.as_ref(), config, &result) {
                Ok(()) => log::info!("bnb stats appended to {path}"),
                Err(e) => log::error!("failed to append bnb stats CSV {path}: {e}"),
            }
        }
        return result;
    }

    log::info!("");

    let n_threads = config.n_threads.max(1);
    log::info!(
        "starting branch-and-bound ({} thread{}): var_type={:?}, n_vars={}, n_terms={}",
        n_threads,
        if n_threads == 1 { "" } else { "s" },
        kernel_instance.var_type(),
        kernel_instance.n_vars(),
        kernel_instance.n_terms()
    );

    // Get the Arc upfront so we can clone it into threads later without cloning the whole instance.
    let instance = Arc::new(kernel_instance);

    // ── Symmetry detection and static symmetry-breaking cuts ─────────────────
    log::info!("Detecting symmetry...");
    {
        let sym_timeout_secs = config
            .time_limit
            .map(|t| (t * 0.1).clamp(1.0, 30.0))
            .unwrap_or(30.0);
        let instance_for_sym = Arc::clone(&instance);
        let (sym_tx, sym_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = sym_tx.send(symmetry::detect_permutation_symmetries(
                instance_for_sym.as_ref(),
            ));
        });
        let permutations = match sym_rx
            .recv_timeout(std::time::Duration::from_secs_f64(sym_timeout_secs))
        {
            Ok(result) => result.unwrap_or_default(),
            Err(_) => {
                log::warn!("symmetry detection timed out after {sym_timeout_secs:.1}s, skipping");
                Vec::new()
            }
        };

        let n_lex_cuts =
            add_lex_comparison_from_permutations(&mut constraint_handler, &permutations);

        if n_lex_cuts > 0 {
            log::info!(
                "Added {} lex-order symmetry-breaking constraint{}",
                n_lex_cuts,
                if n_lex_cuts == 1 { "" } else { "s" }
            );
        } else {
            log::info!("No permutation symmetries detected");
        }
    }

    // ─ Root node initialization and lower bound ───────────────────────────────
    let mut root_node = Node::root(Arc::clone(&instance), C::min_value());

    warmstart_lower_bound(&mut root_node, config, &instance);

    // ── Root node lower bound computation ────────────────────────────────────
    // Computed before warm-start so the table shows the LB first; the warm-start
    // improvements are then logged as subsequent rows.
    let root_lb_elapsed = {
        let new_lb = compute_node_lb(&mut root_node, config, None, &instance);
        root_node.set_lower_bound(new_lb);
        start.elapsed().as_secs_f64()
    };

    // ── Initial solution projection ──────────────────────────────────────────
    // Project any user-supplied solution from original space to kernel space.
    let initial: Option<(C, BitSolution)> = initial_sol
        .as_ref()
        .filter(|s| s.len() == original_instance.n_vars())
        .map(|s| {
            let mut kernel_vals = vec![C::zero(); instance.n_vars()];
            for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
                kernel_vals[new_idx] = s[old_idx];
            }
            let b = BitSolution::from_vec(&kernel_vals);
            let obj = b.evaluate(instance.as_ref());
            log::info!("warm-start (initial solution) objective = {obj}");
            (obj, b)
        });

    // ── Heuristic warm-start ─────────────────────────────────────────────────
    // Runs quietly after the root LB is known; per-heuristic improvements are
    // surfaced as table rows inside run_serial / run_parallel.
    let (warm_incumbent, ws_improvements) = run_warm_start(&instance, config, initial, start);

    // ── Root-level cut generation ─────────────────────────────────────────────
    // Run propagate_by_incumbent once on the unmodified root node (no variables
    // fixed yet).  Every cover and parity cut derived here is valid for the
    // entire search space, so we lift them from the node's local constraint
    // store into the shared constraint_handler.  All BnB nodes then propagate
    // them for free instead of rediscovering the same cuts independently.
    if let Some((inc_obj, _, _)) = warm_incumbent.as_ref() {
        propagate_by_incumbent(&instance, &mut root_node, *inc_obj, &constraint_handler);
        let n_before = constraint_handler.constraints.len();
        for cut in root_node.local_constraints.constraints.drain(..) {
            constraint_handler.add_constraint(cut);
        }
        let n_added = constraint_handler.constraints.len() - n_before;
        if n_added > 0 {
            log::info!(
                "root-level cut generation: added {} global cut{}",
                n_added,
                if n_added == 1 { "" } else { "s" }
            );
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────
    let outcome: SearchOutcome<C> = if n_threads <= 1 {
        run_serial(
            Arc::clone(&instance),
            constraint_handler,
            config,
            start,
            root_node,
            root_lb_elapsed,
            warm_incumbent,
            &ws_improvements,
            solution_receiver,
        )
    } else {
        run_parallel(
            Arc::clone(&instance),
            constraint_handler,
            config,
            start,
            root_node,
            root_lb_elapsed,
            warm_incumbent,
            &ws_improvements,
            n_threads,
            solution_receiver,
        )
    };

    let solving_time = start.elapsed().as_secs_f64();

    // ── Lift kernel solution back to original variable space ─────────────────
    let lifted_solution = outcome
        .incumbent_sol
        .as_ref()
        .map(|kernel_sol| lift_solution(kernel_sol, &new_to_old, &root_node_pre.fixed));

    let objective = lifted_solution
        .as_ref()
        .map(|b| b.evaluate(original_instance))
        .or(outcome.incumbent_obj);

    let result = SolveResult {
        status: outcome.status,
        objective,
        best_bound: outcome.best_bound,
        solution: lifted_solution,
        solving_time,
        tts: outcome.tts,
        n_nodes: outcome.explored,
        pruned_nodes: outcome.pruned,
        unexplored_nodes: outcome.unexplored,
    };

    match (&result.solution, result.objective) {
        (Some(sol), Some(obj)) => log::info!(
            "lifted solution: obj = {obj}, best_bound = {}, solution = {}",
            result.best_bound,
            sol.format_string(instance.as_ref().var_type())
        ),
        (None, _) => log::info!("lifted solution: none, best_bound = {}", result.best_bound),
        (Some(sol), None) => log::info!(
            "lifted solution: obj = n/a, best_bound = {}, solution = {}",
            result.best_bound,
            sol.format_string(instance.as_ref().var_type())
        ),
    }

    if let Some(ref path) = config.solution_file {
        match result.write_solution_file(path, instance.as_ref().var_type()) {
            Ok(()) => log::info!("solution written to {path}"),
            Err(e) => log::error!("failed to write solution file {path}: {e}"),
        }
    }
    if let Some(ref path) = config.stats_csv {
        match append_stats_csv(path, instance.as_ref(), config, &result) {
            Ok(()) => log::info!("bnb stats appended to {path}"),
            Err(e) => log::error!("failed to append bnb stats CSV {path}: {e}"),
        }
    }
    if kernel_report.final_n_vars < kernel_report.initial_n_vars {
        log::info!(
            "kernelized BnB solved reduced problem with {} variables removed",
            kernel_report.initial_n_vars - kernel_report.final_n_vars
        );
    }
    result
}

// ── Serial path ────────────────────────────────────────────────────────────

pub(super) fn run_serial<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    instance: Arc<HuboInstance<C, V>>,
    constraint_handler: ConstraintHandler,
    config: &Config<Lb>,
    start: Instant,
    root_node: Node<C>,
    root_lb_elapsed: f64,
    warm_incumbent: Option<(C, BitSolution, f64)>,
    ws_improvements: &[(&'static str, C, f64)],
    solution_receiver: &mpsc::Receiver<Vec<C>>,
) -> SearchOutcome<C> {
    let mut state = SearchState {
        instance,
        constraint_handler,
        config,
        start,
        root_lb: root_node.lb,
        cached_global_lb: root_node.lb,
        incumbent_obj: None,
        incumbent_sol: None,
        tts: None,
        explored_nodes: 0,
        pruned_nodes: 0,
        leaf_nodes: 0,
        enum_gc_nodes: 0,
        last_log_nodes: 0,
        last_logged_lb: root_node.lb,
        solution_receiver,
        stop_status: None,
    };

    log_table_header();

    let root_lb = root_node.lb;

    log::info!(target: "table",
        "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
        "root-lb", 0u64, 1usize, 0u64, 0u64,
        format!("{:>14}", "n/a"), fmt_coeff(root_lb, 14), "    n/a", root_lb_elapsed
    );

    // Emit one table row per heuristic that improved the incumbent so the
    // warm-start progression is visible at the root before B&B begins.
    for &(name, obj, time) in ws_improvements {
        log::info!(target: "table",
            "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
            name, 0u64, 1usize, 0u64, 0u64,
            fmt_coeff(obj, 14), fmt_coeff(root_lb, 14), format_gap(Some(obj), root_lb), time
        );
    }

    if let Some((obj, bitsol, tts)) = warm_incumbent {
        state.tts = Some(tts);
        state.incumbent_obj = Some(obj);
        state.incumbent_sol = Some(bitsol);
    }
    state.maybe_import_live_solution(&BinaryHeap::new(), None);

    let mut frontier: BinaryHeap<Node<C>> = BinaryHeap::from([root_node]);
    let completed = state.run_loop(&mut frontier);

    let status = if completed {
        Status::Optimal
    } else {
        state.stop_status.unwrap_or(Status::TimeLimit)
    };

    state.refresh_global_lb(&frontier, None);
    let best_bound = if status == Status::Optimal {
        state.incumbent_obj.unwrap_or(root_lb)
    } else {
        state.cached_global_lb
    };

    log_table_row(&state, "final", best_bound, frontier.len());
    log_table_footer();

    log::info!(
        "branch-and-bound finished: status={:?}, time={:.3}s, explored={}, \
         unexplored={}, pruned={}, best_bound={}",
        status,
        start.elapsed().as_secs_f64(),
        state.explored_nodes,
        frontier.len(),
        state.pruned_nodes,
        best_bound,
    );

    if state.enum_gc_nodes > 0 {
        log::info!(
            "small-subproblem enumeration: gray-code (≤{} vars) = {} subproblems",
            super::enumerate::GRAY_CODE_THRESHOLD,
            state.enum_gc_nodes,
        );
    }

    SearchOutcome {
        status,
        explored: state.explored_nodes,
        pruned: state.pruned_nodes,
        unexplored: frontier.len() as u64,
        best_bound,
        incumbent_obj: state.incumbent_obj,
        incumbent_sol: state.incumbent_sol,
        tts: state.tts,
    }
}

fn solve_decomposed<C: Coeff, V: CoverCutDomain, Lb: LowerBound + Clone>(
    original_instance: &Arc<HuboInstance<C, V>>,
    splits: Vec<symmetry::ComponentSplit<C, V>>,
    new_to_old: &[usize],
    fixes: &Fixes,
    config: &Config<Lb>,
) -> SolveResult<C> {
    let n_kernel_vars = new_to_old.len();
    let mut kernel_bits = vec![false; n_kernel_vars];

    let mut total_bound = C::zero();
    let mut overall_status = Status::Optimal;
    let mut total_nodes = 0u64;
    let mut total_pruned = 0u64;
    let mut total_unexplored = 0u64;
    let mut total_solving_time = 0.0f64;
    let mut has_solution = true;

    let mut sub_config = config.clone();
    sub_config.solution_file = None;
    sub_config.stats_csv = None;

    for split in splits {
        let (_, sub_rx) = mpsc::channel();
        let sub_arc = Arc::new(split.sub_instance);
        let sub_result = solve(&sub_arc, &sub_config, None, &sub_rx);

        total_solving_time += sub_result.solving_time;
        total_nodes += sub_result.n_nodes;
        total_pruned += sub_result.pruned_nodes;
        total_unexplored += sub_result.unexplored_nodes;
        total_bound += sub_result.best_bound;

        if sub_result.status != Status::Optimal {
            overall_status = sub_result.status;
        }

        match sub_result.solution {
            Some(ref sub_sol) => {
                for (sub_idx, &kernel_idx) in split.new_to_old.iter().enumerate() {
                    kernel_bits[kernel_idx] = sub_sol.values.contains(sub_idx);
                }
            }
            None => has_solution = false,
        }
    }

    let (solution, objective) = if has_solution {
        let kernel_sol = BitSolution::from_bool_vec(kernel_bits);
        let lifted = lift_solution(&kernel_sol, new_to_old, fixes);
        let obj = lifted.evaluate(original_instance);
        (Some(lifted), Some(obj))
    } else {
        (None, None)
    };

    SolveResult {
        status: overall_status,
        objective,
        best_bound: total_bound,
        solution,
        solving_time: total_solving_time,
        tts: None,
        n_nodes: total_nodes,
        pruned_nodes: total_pruned,
        unexplored_nodes: total_unexplored,
    }
}

/// Lift a solution from the kernel (reduced) variable space back to the
/// original variable space using the `new_to_old` index map produced by
/// `HuboInstance::apply_fixes`.
///
/// For each original variable:
/// - Free variables take their value from the kernel solution via `new_to_old`.
/// - Fixed variables take their assigned value from `fixes`.
fn lift_solution(kernel_sol: &BitSolution, new_to_old: &[usize], fixes: &Fixes) -> BitSolution {
    let n_vars = fixes.assigned.len();
    let mut full = BitSolution::new(n_vars);
    for old_idx in 0..n_vars {
        if let Some(high) = fixes.get(old_idx) {
            full.values.set(old_idx, high);
        }
    }
    for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
        full.values
            .set(old_idx, kernel_sol.values.contains(new_idx));
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lift_solution_uses_new_to_old_mapping() {
        let mut fixes = Fixes::new(5);
        fixes.set(1, true).unwrap();
        fixes.set(3, false).unwrap();

        // Kernel variables 0, 1, 2 correspond to original variables 0, 2, 4.
        let new_to_old = vec![0, 2, 4];
        let kernel_sol = BitSolution::from_bool_vec(vec![true, false, true]);

        let lifted = lift_solution(&kernel_sol, &new_to_old, &fixes);

        assert_eq!(
            (0..5)
                .map(|idx| lifted.values.contains(idx))
                .collect::<Vec<_>>(),
            vec![true, true, false, false, true]
        );
    }
}
