use std::collections::HashSet;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};

use crate::solver::bnb::util::{fmt_coeff, format_gap, log_table_footer, log_table_header};

use super::*;

// ── Shared incumbent ──────────────────────────────────────────────────────────

pub(super) struct SharedIncumbent<C: Coeff> {
    pub(crate) obj: Option<C>,
    pub(crate) sol: Option<BitSolution>,
    pub(crate) tts: Option<f64>,
}

// ── Shadow LB tracker: min-heap with lazy deletion ───────────────────────────
//
// Every node that is queued (in the global injector or a worker's local stack)
// has an entry in here.  Nodes that are being actively processed by a worker
// are removed from the tracker and instead recorded in `active_lbs`.
//
// Global LB = min(tracker.min_lb(), min(active_lbs))
//
// This is provably monotone: children always have LB ≥ parent, so inserting
// children and removing the parent never decreases the minimum.

struct LbEntry<C: Coeff> {
    lb: C,
    id: u64,
}

impl<C: Coeff> PartialEq for LbEntry<C> {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl<C: Coeff> Eq for LbEntry<C> {}
impl<C: Coeff> PartialOrd for LbEntry<C> {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl<C: Coeff> Ord for LbEntry<C> {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // Reversed → smaller lb floats to the top (min-heap behaviour).
        o.lb.partial_cmp(&self.lb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(self.id.cmp(&o.id))
    }
}

struct LbTracker<C: Coeff> {
    heap: BinaryHeap<LbEntry<C>>,
    /// IDs logically removed but not yet cleaned from the heap.
    removed: HashSet<u64>,
    next_id: u64,
}

impl<C: Coeff> LbTracker<C> {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            removed: HashSet::new(),
            next_id: 0,
        }
    }

    fn push(&mut self, lb: C) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.heap.push(LbEntry { lb, id });
        id
    }

    fn remove(&mut self, id: u64) {
        self.removed.insert(id);
    }

    /// Return the minimum LB of all live (non-removed) entries, cleaning up
    /// stale heap tops as a side-effect.
    fn min_lb(&mut self) -> Option<C> {
        loop {
            match self.heap.peek() {
                None => return None,
                Some(e) if self.removed.contains(&e.id) => {
                    let e = self.heap.pop().unwrap();
                    self.removed.remove(&e.id);
                }
                Some(e) => return Some(e.lb),
            }
        }
    }
}

// ── TrackedNode: node paired with its LB-tracker registration ────────────────

struct TrackedNode<C: Coeff> {
    node: Node<C>,
    tracker_id: u64,
}

// ── Shared state ──────────────────────────────────────────────────────────────

struct WorkStealShared<C: Coeff, V: VarDomain, Lb> {
    instance: Arc<HuboInstance<C, V>>,
    constraint_handler: Arc<ConstraintHandler>,
    config: Arc<Config<Lb>>,
    injector: Injector<TrackedNode<C>>,
    incumbent: Mutex<SharedIncumbent<C>>,
    stop: AtomicBool,
    stop_status: Mutex<Option<Status>>,
    start: Instant,
    idle: Mutex<usize>,
    idle_cond: Condvar,
    n_workers: usize,
    /// All queued nodes are registered here; popped nodes are removed.
    lb_tracker: Mutex<LbTracker<C>>,
    /// Per-worker LB of the node currently being processed (None = idle).
    active_lbs: Vec<Mutex<Option<C>>>,
    /// Total children pushed (seeds + descendants); used for frontier size estimate.
    total_pushed: AtomicU64,
    explored: AtomicU64,
    pruned: AtomicU64,
    leaves: AtomicU64,
    enum_gc: AtomicU64,
}

impl<C: Coeff, V: VarDomain, Lb: LowerBound> WorkStealShared<C, V, Lb> {
    fn incumbent_obj(&self) -> Option<C> {
        self.incumbent.lock().unwrap().obj
    }

    fn try_update_incumbent(&self, objective: C, bitsol: BitSolution) -> bool {
        let mut inc = self.incumbent.lock().unwrap();
        let improved = inc.obj.is_none_or(|best| objective < best);
        if improved {
            inc.obj = Some(objective);
            inc.sol = Some(bitsol);
            if inc.tts.is_none() {
                inc.tts = Some(self.start.elapsed().as_secs_f64());
            }
        }
        improved
    }

    /// Exact global lower bound, monotone non-decreasing over the search.
    ///
    /// = min(queued node LBs in tracker, active node LBs)
    /// Falls back to `fallback` when both are empty (search exhausted).
    fn global_lb(&self, fallback: C) -> C {
        let tracker_min = self.lb_tracker.lock().unwrap().min_lb();
        let active_min = self
            .active_lbs
            .iter()
            .filter_map(|m| *m.lock().unwrap())
            .reduce(|a, b| if b < a { b } else { a });
        tracker_min
            .into_iter()
            .chain(active_min)
            .reduce(|a, b| if b < a { b } else { a })
            .unwrap_or(fallback)
    }

    fn approx_frontier(&self) -> u64 {
        self.total_pushed
            .load(Ordering::Relaxed)
            .saturating_sub(self.explored.load(Ordering::Relaxed))
    }
}

// ── RAII: clear active_lb slot on drop ───────────────────────────────────────

struct ActiveLbGuard<'a, C: Coeff>(&'a Mutex<Option<C>>);

impl<C: Coeff> Drop for ActiveLbGuard<'_, C> {
    fn drop(&mut self) {
        *self.0.lock().unwrap() = None;
    }
}

// ── Work-finding helper ───────────────────────────────────────────────────────

fn find_task<C: Coeff>(
    local: &Worker<TrackedNode<C>>,
    injector: &Injector<TrackedNode<C>>,
    stealers: &[Stealer<TrackedNode<C>>],
) -> Option<TrackedNode<C>> {
    local.pop().or_else(|| {
        loop {
            match injector.steal_batch_and_pop(local) {
                Steal::Success(t) => return Some(t),
                Steal::Empty => break,
                Steal::Retry => {}
            }
        }
        stealers.iter().find_map(|s| {
            loop {
                match s.steal() {
                    Steal::Success(t) => return Some(t),
                    Steal::Empty => return None,
                    Steal::Retry => {}
                }
            }
        })
    })
}

// ── Phase 1: cheap tree expansion to seed parallel workers ───────────────────
//
// Expands the root to ~n_seeds open nodes using only cheap per-node operations
// (constraint propagation and heuristic branching).  Kernelization, probing,
// and the LB oracle are intentionally skipped here; workers run them in full
// during Phase 2.  Progress is logged to the shared table.

fn expand_phase<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    instance: &Arc<HuboInstance<C, V>>,
    constraint_handler: &ConstraintHandler,
    config: &Config<Lb>,
    root: Node<C>,
    n_seeds: usize,
    start: Instant,
    incumbent: &mut Option<C>,
    incumbent_sol: &mut Option<BitSolution>,
) -> Vec<Node<C>> {
    let root_lb = root.lb;
    let mut open: BinaryHeap<Node<C>> = BinaryHeap::new();
    open.push(root);

    let mut explored = 0u64;
    let mut pruned = 0u64;
    let mut leaves = 0u64;
    let log_every = config.progress_every_nodes.unwrap_or(5000).max(1);
    let mut last_log_explored = 0u64;
    let mut last_inc = *incumbent;
    let mut last_logged_lb = root_lb;

    macro_rules! log_expand_row {
        ($event:expr, $bound:expr) => {{
            let elapsed = start.elapsed().as_secs_f64();
            let inc_str = incumbent
                .map(|v| fmt_coeff(v, 14))
                .unwrap_or_else(|| format!("{:>14}", "n/a"));
            log::info!(target: "table",
                "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
                $event,
                explored,
                open.len(),
                pruned,
                leaves,
                inc_str,
                fmt_coeff($bound, 14),
                format_gap(*incumbent, $bound),
                elapsed
            );
        }};
    }

    while open.len() < n_seeds {
        let Some(mut node) = open.pop() else { break };

        if crate::interrupt::is_interrupted() {
            open.push(node);
            break;
        }

        if incumbent
            .is_some_and(|best: C| node.lb.to_f64() + config.optimality_tol >= best.to_f64())
        {
            pruned += 1;
            continue;
        }

        explored += 1;

        if !propagate_constraints_only(instance, constraint_handler, &mut node) {
            pruned += 1;
            continue;
        }

        if let Some(best) = *incumbent
            && !propagate_by_incumbent(instance, &mut node, best, constraint_handler)
        {
            pruned += 1;
            continue;
        }

        // No kernelization, probing, or LB oracle — workers handle those in Phase 2.
        match select_branch_var(
            &node,
            instance,
            constraint_handler,
            None,
            config,
            *incumbent,
        ) {
            BranchChoice::On(var) => {
                for high in [true, false] {
                    let mut child = node.child(instance, var, high);
                    if !propagate_constraints_only(instance, constraint_handler, &mut child) {
                        pruned += 1;
                        continue;
                    }
                    child.term_by_free_vars = None;
                    open.push(child);
                }
            }
            BranchChoice::Leaf => {
                leaves += 1;
                let bitsol = node.to_bitsolution(instance);
                let obj = bitsol.evaluate(instance.as_ref());
                if incumbent.is_none_or(|best: C| obj < best) {
                    *incumbent = Some(obj);
                    *incumbent_sol = Some(bitsol);
                }
            }
            BranchChoice::Infeasible => {
                pruned += 1;
            }
        }

        let bound = open.peek().map(|n| n.lb).unwrap_or(root_lb);

        let inc_changed = *incumbent != last_inc;
        let bound_improved = {
            let new_lb = bound.to_f64();
            let old_lb = last_logged_lb.to_f64();
            if new_lb <= old_lb {
                false
            } else {
                let thr = config.bound_log_min_improvement_pct;
                if thr <= 0.0 {
                    true
                } else if let Some(ub) = *incumbent {
                    let uf = ub.to_f64();
                    let gap = |lb: f64| {
                        let d = uf.abs().max(lb.abs()).max(1e-12);
                        ((uf - lb).max(0.0) / d) * 100.0
                    };
                    gap(old_lb) - gap(new_lb) >= thr
                } else {
                    false
                }
            }
        };
        let progress_due = explored.saturating_sub(last_log_explored) >= log_every;

        if inc_changed || bound_improved || progress_due {
            last_inc = *incumbent;
            last_log_explored = explored;
            if inc_changed || bound_improved {
                last_logged_lb = bound;
            }
            let event = if inc_changed {
                "expand-inc"
            } else if bound_improved {
                "bound"
            } else {
                "expand"
            };
            log_expand_row!(event, bound);
        }
    }

    open.into_vec()
}

// ── Phase 2: per-node processing ──────────────────────────────────────────────

fn process_node<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    tracked: TrackedNode<C>,
    local: &Worker<TrackedNode<C>>,
    shared: &WorkStealShared<C, V, Lb>,
    worker_id: usize,
) {
    let mut node = tracked.node;

    // Move from tracker (queued) → active_lbs (being processed).
    shared.lb_tracker.lock().unwrap().remove(tracked.tracker_id);
    let _lb_guard = ActiveLbGuard(&shared.active_lbs[worker_id]);
    *shared.active_lbs[worker_id].lock().unwrap() = Some(node.lb);

    shared.explored.fetch_add(1, Ordering::Relaxed);

    let tol = shared.config.optimality_tol;
    // Snapshot the incumbent; re-read before each major step to pick up
    // improvements from other workers without locking on every prune check.
    let mut ub = shared.incumbent_obj();

    if ub.is_some_and(|best| node.lb.to_f64() + tol >= best.to_f64()) {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    kernelize_search_node(&shared.instance, &mut node, &shared.config, ub);

    // Roof duality may set node.lb = max_value to signal infeasibility.
    if ub.is_some_and(|best| node.lb.to_f64() + tol >= best.to_f64()) {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    if !propagate_constraints_only(
        &shared.instance,
        shared.constraint_handler.as_ref(),
        &mut node,
    ) {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    ub = shared.incumbent_obj();
    if let Some(best) = ub
        && !propagate_by_incumbent(
            &shared.instance,
            &mut node,
            best,
            shared.constraint_handler.as_ref(),
        )
    {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    ub = shared.incumbent_obj();
    if !probe_node(
        &shared.instance,
        &mut node,
        shared.constraint_handler.as_ref(),
        ub,
        &shared.config.probing,
    ) {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    // Final lb computation after all propagations and probing; tightest possible bound.
    ub = shared.incumbent_obj();
    let (new_lb, spin_core) =
        compute_node_lb_with_core(&mut node, &shared.config, ub, &shared.instance);
    node.set_lower_bound(new_lb);
    *shared.active_lbs[worker_id].lock().unwrap() = Some(node.lb);

    if ub.is_some_and(|best| node.lb.to_f64() + tol >= best.to_f64()) {
        shared.pruned.fetch_add(1, Ordering::Relaxed);
        return;
    }

    // Small-subproblem dispatch: solve inline by enumeration when the
    // residual problem is small enough (mirrors the serial path).
    let n_free = node.fixed.num_free();
    if n_free > 0 && n_free <= enumerate::GRAY_CODE_THRESHOLD {
        let problem = enumerate::LocalProblem::build(shared.instance.as_ref(), &node);
        let var_type = shared.instance.var_type();
        // Workers are already parallel threads — calling gray_code_solve_parallel
        // here would compete with other workers for the rayon thread pool.
        let (_best_obj, best_pattern) = enumerate::gray_code_solve(&problem, var_type);
        shared.enum_gc.fetch_add(1, Ordering::Relaxed);

        shared.leaves.fetch_add(1, Ordering::Relaxed);
        let mut values = node.fixed.values.clone();
        for (i, &g) in problem.local_to_global.iter().enumerate() {
            values.set(g, (best_pattern >> i) & 1 == 1);
        }
        let bitsol = BitSolution { values };
        let objective = bitsol.evaluate(shared.instance.as_ref());
        let _ = shared.try_update_incumbent(objective, bitsol);
        return;
    }

    // Refresh before branching: a tighter bound means better variable selection
    // (strong branching) and more aggressive child pruning.
    ub = shared.incumbent_obj();
    match select_branch_var(
        &node,
        &shared.instance,
        shared.constraint_handler.as_ref(),
        spin_core,
        &shared.config,
        ub,
    ) {
        BranchChoice::On(var) => {
            let mut children: Vec<Node<C>> = Vec::with_capacity(2);

            for high in [true, false] {
                let mut child = node.child(&shared.instance, var, high);
                if !propagate_constraints_only(
                    &shared.instance,
                    shared.constraint_handler.as_ref(),
                    &mut child,
                ) {
                    shared.pruned.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let new_lb = compute_node_lb(&mut child, &shared.config, ub, &shared.instance);
                child.set_lower_bound(new_lb);
                if ub.is_some_and(|best| child.lb.to_f64() + tol >= best.to_f64()) {
                    shared.pruned.fetch_add(1, Ordering::Relaxed);
                } else {
                    child.term_by_free_vars = None;
                    children.push(child);
                }
            }

            if !children.is_empty() {
                // Sort descending: children[0] = highest lb (worst), children[last] = lowest lb (best).
                children.sort_unstable_by(|a, b| {
                    b.lb.partial_cmp(&a.lb).unwrap_or(std::cmp::Ordering::Equal)
                });
                let n = children.len() as u64;
                // Route:
                // · best child (lowest lb, last after sort) → worker's local LIFO so it
                //   continues DFS on the most promising branch with a plain local pop,
                //   avoiding any contention on the shared injector.
                // · all other children → global injector for other workers to steal.
                // Seeds are given accurate lbs before workers start, so the tracker
                // always holds true lower bounds and lb convergence is maintained.
                let best = children.pop();
                let mut tracked_rest: Vec<TrackedNode<C>> = Vec::with_capacity(children.len());
                let tracked_best: Option<TrackedNode<C>>;
                {
                    let mut tracker = shared.lb_tracker.lock().unwrap();
                    for child in children {
                        let tid = tracker.push(child.lb);
                        tracked_rest.push(TrackedNode {
                            node: child,
                            tracker_id: tid,
                        });
                    }
                    tracked_best = best.map(|child| {
                        let tid = tracker.push(child.lb);
                        TrackedNode {
                            node: child,
                            tracker_id: tid,
                        }
                    });
                }
                for tc in tracked_rest {
                    shared.injector.push(tc);
                }
                if let Some(tc) = tracked_best {
                    local.push(tc);
                }
                shared.total_pushed.fetch_add(n, Ordering::Relaxed);
                shared.idle_cond.notify_one();
            }
        }
        BranchChoice::Leaf => {
            shared.leaves.fetch_add(1, Ordering::Relaxed);
            let bitsol = node.to_bitsolution(&shared.instance);
            let obj = bitsol.evaluate(shared.instance.as_ref());
            let _ = shared.try_update_incumbent(obj, bitsol);
        }
        BranchChoice::Infeasible => {
            shared.pruned.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ── Phase 2: worker loop ──────────────────────────────────────────────────────

fn worker_run<C: Coeff, V: CoverCutDomain, Lb: LowerBound>(
    local: Worker<TrackedNode<C>>,
    stealers: Arc<Vec<Stealer<TrackedNode<C>>>>,
    shared: Arc<WorkStealShared<C, V, Lb>>,
    worker_id: usize,
) {
    'main: loop {
        while let Some(tracked) = find_task(&local, &shared.injector, &stealers) {
            if shared.stop.load(Ordering::Acquire) {
                return;
            }
            // Mirror the serial solver's per-node stopping-criteria check so
            // the time limit and interrupt signal are respected within one node
            // processing time rather than within one coordinator sleep cycle.
            // Use swap so only the first worker to detect the condition pays
            // the cost of locking stop_status.
            let early_stop = if crate::interrupt::is_interrupted() {
                Some(Status::Interrupted)
            } else if shared
                .config
                .time_limit
                .is_some_and(|l| shared.start.elapsed().as_secs_f64() >= l)
            {
                Some(Status::TimeLimit)
            } else {
                None
            };
            if let Some(status) = early_stop {
                if !shared.stop.swap(true, Ordering::AcqRel) {
                    *shared.stop_status.lock().unwrap() = Some(status);
                }
                shared.idle_cond.notify_all();
                return;
            }
            process_node(tracked, &local, &shared, worker_id);
        }

        // No work found. Lock before rechecking to close the race window
        // between "saw nothing" and a concurrent push + notify.
        let mut idle = shared.idle.lock().unwrap();

        if let Some(tracked) = find_task(&local, &shared.injector, &stealers) {
            drop(idle);
            if shared.stop.load(Ordering::Acquire) {
                return;
            }
            process_node(tracked, &local, &shared, worker_id);
            continue 'main;
        }

        *idle += 1;
        if *idle == shared.n_workers {
            shared.stop.store(true, Ordering::Release);
            shared.idle_cond.notify_all();
            return;
        }

        loop {
            idle = shared
                .idle_cond
                .wait_timeout(idle, Duration::from_millis(5))
                .unwrap()
                .0;

            if shared.stop.load(Ordering::Acquire) {
                return;
            }

            if let Some(tracked) = find_task(&local, &shared.injector, &stealers) {
                *idle -= 1;
                drop(idle);
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                process_node(tracked, &local, &shared, worker_id);
                continue 'main;
            }
        }
    }
}

// ── run_parallel ──────────────────────────────────────────────────────────────

const SEEDS_PER_THREAD: usize = 4;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_parallel<C: Coeff, V: CoverCutDomain, Lb: LowerBound + Clone>(
    instance: Arc<HuboInstance<C, V>>,
    constraint_handler: ConstraintHandler,
    config: &Config<Lb>,
    start: Instant,
    root_node: Node<C>,
    root_lb_elapsed: f64,
    warm_incumbent: Option<(C, BitSolution, f64)>,
    ws_improvements: &[(&'static str, C, f64)],
    n_threads: usize,
    solution_receiver: &mpsc::Receiver<Vec<C>>,
) -> SearchOutcome<C> {
    let warm_incumbent_obj = warm_incumbent.as_ref().map(|(o, _, _)| *o);
    let warm_incumbent_tts = warm_incumbent.as_ref().map(|(_, _, t)| *t);
    let root_lb = root_node.lb;

    log_table_header();

    // Log the root's lower bound as the first table entry.  At this logical
    // point warm-start has not yet run, so the incumbent column is n/a.
    log::info!(target: "table",
        "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
        "root-lb", 0u64, 1u64, 0u64, 0u64,
        format!("{:>14}", "n/a"), fmt_coeff(root_lb, 14), "    n/a", root_lb_elapsed
    );

    // Emit one table row per heuristic that improved the incumbent so the
    // warm-start progression is visible at the root before B&B begins.
    for &(name, obj, time) in ws_improvements {
        log::info!(target: "table",
            "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
            name, 0u64, 1u64, 0u64, 0u64,
            fmt_coeff(obj, 14), fmt_coeff(root_lb, 14), format_gap(Some(obj), root_lb), time
        );
    }

    // ── Fast path: gray-code enumeration ─────────────────────────────────
    // n_free ≤ 25:  serial walk, no threads spawned.
    // 25 < n_free ≤ 25 + k_bits:  fix k = n_free − 25 prefix variables so
    //   each of the 2^k rayon tasks walks exactly 25 tail variables (2^25
    //   iterations).  We enter only when 2^k ≤ n_threads.
    let k_bits = n_threads.ilog2() as usize; // floor(log2(n_threads))
    let par_enum_threshold = enumerate::GRAY_CODE_THRESHOLD + k_bits;
    let n_free = instance.n_vars();
    if n_free > 0 && n_free <= par_enum_threshold {
        let problem = enumerate::LocalProblem::build(instance.as_ref(), &root_node);
        let (_, enum_pattern) = if n_free <= enumerate::GRAY_CODE_THRESHOLD {
            enumerate::gray_code_solve(&problem, instance.var_type())
        } else {
            let k = n_free - enumerate::GRAY_CODE_THRESHOLD;
            enumerate::gray_code_solve_parallel(&problem, instance.var_type(), k)
        };

        let mut values = root_node.fixed.values.clone();
        for (i, &g) in problem.local_to_global.iter().enumerate() {
            values.set(g, (enum_pattern >> i) & 1 == 1);
        }
        let enum_sol = BitSolution { values };
        let enum_obj = enum_sol.evaluate(instance.as_ref());

        let (best_obj, best_sol, tts) = match warm_incumbent_obj {
            Some(w) if w <= enum_obj => (
                Some(w),
                warm_incumbent.as_ref().map(|(_, s, _)| s.clone()),
                warm_incumbent_tts,
            ),
            _ => (
                Some(enum_obj),
                Some(enum_sol),
                Some(start.elapsed().as_secs_f64()),
            ),
        };

        let elapsed = start.elapsed().as_secs_f64();
        let best_bound = best_obj.unwrap_or(root_lb);
        let event = if n_free <= enumerate::GRAY_CODE_THRESHOLD {
            "enum-serial"
        } else {
            "enum-par"
        };
        log::info!(target: "table",
            "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
            event, 0u64, 0u64, 0u64, 1u64,
            best_obj.map(|v| fmt_coeff(v, 14)).unwrap_or_else(|| format!("{:>14}", "n/a")),
            fmt_coeff(best_bound, 14),
            format_gap(best_obj, best_bound),
            elapsed
        );
        log_table_footer();
        if n_free <= enumerate::GRAY_CODE_THRESHOLD {
            log::info!(
                "serial gray-code enumeration: {} vars ({} assignments)",
                n_free,
                1usize << n_free
            );
        } else {
            let k = n_free - enumerate::GRAY_CODE_THRESHOLD;
            log::info!(
                "parallel gray-code enumeration: {} vars, k={} ({} tasks, {} threads)",
                n_free,
                k,
                1usize << k,
                n_threads
            );
        }

        return SearchOutcome {
            status: Status::Optimal,
            explored: 0,
            pruned: 0,
            unexplored: 0,
            best_bound,
            incumbent_obj: best_obj,
            incumbent_sol: best_sol,
            tts,
        };
    }

    // ── Phase 1: cheap tree expansion to seed workers ────────────────────
    let n_seeds_target = n_threads * SEEDS_PER_THREAD;
    log::debug!(
        "parallel BnB: expanding to {} seed nodes ({} × {})",
        n_seeds_target,
        n_threads,
        SEEDS_PER_THREAD
    );

    let mut phase1_incumbent = warm_incumbent_obj;
    let mut phase1_sol = warm_incumbent.as_ref().map(|(_, s, _)| s.clone());

    let mut seed_nodes = expand_phase(
        &instance,
        &constraint_handler,
        config,
        root_node,
        n_seeds_target,
        start,
        &mut phase1_incumbent,
        &mut phase1_sol,
    );

    log::debug!(
        "parallel BnB: expansion done — {} seeds, incumbent = {}",
        seed_nodes.len(),
        phase1_incumbent
            .map(|v| v.to_f64().to_string())
            .unwrap_or_else(|| "n/a".to_string()),
    );

    if seed_nodes.is_empty() {
        let status = if crate::interrupt::is_interrupted() {
            Status::Interrupted
        } else {
            Status::Optimal
        };
        let best_bound = phase1_incumbent.unwrap_or(root_lb);
        let inc_str = phase1_incumbent
            .map(|v| fmt_coeff(v, 14))
            .unwrap_or_else(|| format!("{:>14}", "n/a"));
        log::info!(target: "table",
            "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
            "final", 0u64, 0usize, 0u64, 0u64,
            inc_str, fmt_coeff(best_bound, 14),
            format_gap(phase1_incumbent, best_bound),
            start.elapsed().as_secs_f64()
        );
        log_table_footer();
        return SearchOutcome {
            status,
            explored: 0,
            pruned: 0,
            unexplored: 0,
            best_bound,
            incumbent_obj: phase1_incumbent,
            incumbent_sol: phase1_sol,
            tts: if phase1_incumbent.is_some() && phase1_incumbent != warm_incumbent_obj {
                Some(start.elapsed().as_secs_f64())
            } else {
                warm_incumbent_tts
            },
        };
    }

    let n_seeds = seed_nodes.len();

    // ── Compute accurate lower bounds for seeds ───────────────────────────
    // expand_phase branches without running the LB oracle, so every seed
    // inherits lb = root_lb.  If seeds enter the tracker with that stale
    // value, global_lb is pinned at root_lb until every seed has been
    // popped — identical to having no bound progress for the first n_seeds
    // node-processings.  The serial solver calls compute_node_lb on every
    // child before pushing it to the frontier; do the same here so the
    // tracker holds accurate values and the global bound rises immediately.
    for node in &mut seed_nodes {
        let new_lb = compute_node_lb(node, config, phase1_incumbent, &instance);
        node.set_lower_bound(new_lb);
    }

    // ── Build tracker, register seeds, create injector ───────────────────
    let mut lb_tracker = LbTracker::new();
    let injector: Injector<TrackedNode<C>> = Injector::new();

    // Sort seeds descending by lb before pushing.  steal_batch_and_pop moves tasks from
    // the injector (FIFO) into a worker's local LIFO, reversing the order: the last item
    // stolen (from the back of the injector) lands on top of the local LIFO and is
    // processed first.  Pushing the lowest-lb seeds last (back of the injector) therefore
    // ensures they are processed first, driving the global lower bound up quickly.
    seed_nodes
        .sort_unstable_by(|a, b| b.lb.partial_cmp(&a.lb).unwrap_or(std::cmp::Ordering::Equal));
    for node in seed_nodes {
        let tid = lb_tracker.push(node.lb);
        injector.push(TrackedNode {
            node,
            tracker_id: tid,
        });
    }

    // ── Per-thread Worker/Stealer pairs ───────────────────────────────────
    let mut workers: Vec<Worker<TrackedNode<C>>> = Vec::with_capacity(n_threads);
    let mut all_stealers: Vec<Stealer<TrackedNode<C>>> = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let w = Worker::new_lifo();
        all_stealers.push(w.stealer());
        workers.push(w);
    }
    let stealers: Arc<Vec<Stealer<TrackedNode<C>>>> = Arc::new(all_stealers);

    let incumbent_tts = if phase1_incumbent.is_some() && phase1_incumbent != warm_incumbent_obj {
        Some(start.elapsed().as_secs_f64())
    } else {
        warm_incumbent_tts
    };

    let shared = Arc::new(WorkStealShared {
        instance: Arc::clone(&instance),
        constraint_handler: Arc::new(constraint_handler),
        // Workers inherit all search parameters from the parent config.
        // Override only fields that the workers must not act on themselves
        // (logging, solution writing, warm-start, and thread spawning are
        // handled exclusively by the coordinator / run_parallel).
        config: Arc::new(Config {
            progress_every_nodes: None,
            stats_csv: None,
            instance_name: None,
            solution_file: None,
            warm_start_heuristics: false,
            warm_start_heuristic_time_limit: None,
            n_threads: 1,
            ..config.clone()
        }),
        injector,
        incumbent: Mutex::new(SharedIncumbent {
            obj: phase1_incumbent,
            sol: phase1_sol,
            tts: incumbent_tts,
        }),
        stop: AtomicBool::new(false),
        stop_status: Mutex::new(None),
        start,
        idle: Mutex::new(0),
        idle_cond: Condvar::new(),
        n_workers: n_threads,
        lb_tracker: Mutex::new(lb_tracker),
        active_lbs: (0..n_threads).map(|_| Mutex::new(None)).collect(),
        total_pushed: AtomicU64::new(n_seeds as u64),
        explored: AtomicU64::new(0),
        pruned: AtomicU64::new(0),
        leaves: AtomicU64::new(0),
        enum_gc: AtomicU64::new(0),
    });

    // ── Spawn worker threads ──────────────────────────────────────────────
    let handles: Vec<_> = workers
        .into_iter()
        .enumerate()
        .map(|(id, local)| {
            let sh = Arc::clone(&shared);
            let st = Arc::clone(&stealers);
            std::thread::spawn(move || worker_run(local, st, sh, id))
        })
        .collect();

    // ── Coordinator loop ──────────────────────────────────────────────────
    let log_every_nodes = config.progress_every_nodes.unwrap_or(5000).max(1);
    let mut last_log_nodes = 0u64;
    let mut last_logged_incumbent = phase1_incumbent;

    macro_rules! log_row {
        ($event:expr, $elapsed:expr, $bound_override:expr) => {{
            let explored = shared.explored.load(Ordering::Relaxed);
            let pruned = shared.pruned.load(Ordering::Relaxed);
            let leaves = shared.leaves.load(Ordering::Relaxed);
            let frontier = shared.approx_frontier();
            let inc_obj = shared.incumbent.lock().unwrap().obj;
            // global_lb is the exact, monotone lower bound from the shadow tracker.
            let bound: C = $bound_override.unwrap_or_else(|| shared.global_lb(root_lb));

            let inc_str = inc_obj
                .map(|v| fmt_coeff(v, 14))
                .unwrap_or_else(|| format!("{:>14}", "n/a"));
            let bb_str = fmt_coeff(bound, 14);
            let gap = format_gap(inc_obj, bound);
            log::info!(target: "table",
                "│ {:<10} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │ {:>14} │ {:>14} │ {:>8} │ {:>7.3} │",
                $event, explored, frontier, pruned, leaves, inc_str, bb_str, gap, $elapsed
            );
        }};
    }

    loop {
        std::thread::sleep(Duration::from_millis(100));
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        let elapsed = start.elapsed().as_secs_f64();

        let ext: Option<Status> = if crate::interrupt::is_interrupted() {
            Some(Status::Interrupted)
        } else if config.time_limit.is_some_and(|l| elapsed >= l) {
            Some(Status::TimeLimit)
        } else if config
            .node_limit
            .is_some_and(|l| shared.explored.load(Ordering::Relaxed) >= l)
        {
            Some(Status::NodeLimit)
        } else if let Some(cutoff) = config.cutoff {
            shared
                .incumbent
                .lock()
                .unwrap()
                .obj
                .and_then(|obj| (obj.to_f64() <= cutoff).then_some(Status::Cutoff))
        } else {
            None
        };

        if let Some(status) = ext {
            if !shared.stop.swap(true, Ordering::AcqRel) {
                *shared.stop_status.lock().unwrap() = Some(status);
            }
            shared.idle_cond.notify_all();
            break;
        }

        while let Ok(src) = solution_receiver.try_recv() {
            let bitsol = BitSolution::from_vec(&src);
            let obj = bitsol.evaluate(&shared.instance);
            if shared.try_update_incumbent(obj, bitsol) {
                last_logged_incumbent = Some(obj);
                log_row!("injected", elapsed, None::<C>);
            }
        }

        let inc_obj = shared.incumbent.lock().unwrap().obj;
        if inc_obj != last_logged_incumbent {
            last_logged_incumbent = inc_obj;
            log_row!("incumbent", elapsed, None::<C>);
        }

        let explored_now = shared.explored.load(Ordering::Relaxed);
        if explored_now.saturating_sub(last_log_nodes) >= log_every_nodes {
            last_log_nodes = explored_now;
            log_row!("progress", elapsed, None::<C>);
        }
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }

    let explored = shared.explored.load(Ordering::Relaxed);
    let pruned = shared.pruned.load(Ordering::Relaxed);
    let stop_status = shared.stop_status.lock().unwrap().take();

    let status = if stop_status.is_none() {
        Status::Optimal
    } else {
        stop_status.unwrap_or(Status::TimeLimit)
    };

    let inc = shared.incumbent.lock().unwrap();
    let incumbent_obj = inc.obj;
    let incumbent_sol = inc.sol.clone();
    let tts = inc.tts;
    drop(inc);

    let best_bound = if status == Status::Optimal {
        incumbent_obj.unwrap_or(root_lb)
    } else {
        shared.global_lb(root_lb)
    };

    log_row!("final", start.elapsed().as_secs_f64(), Some(best_bound));
    log_table_footer();

    let enum_gc = shared.enum_gc.load(Ordering::Relaxed);
    if enum_gc > 0 {
        log::info!(
            "small-subproblem enumeration: gray-code (≤{} vars) = {} subproblems",
            enumerate::GRAY_CODE_THRESHOLD,
            enum_gc,
        );
    }

    log::info!(
        "parallel BnB finished: status={:?}, time={:.3}s, threads={}, explored={}, pruned={}, best_bound={}",
        status,
        start.elapsed().as_secs_f64(),
        n_threads,
        explored,
        pruned,
        best_bound,
    );

    SearchOutcome {
        status,
        explored,
        pruned,
        unexplored: 0,
        best_bound,
        incumbent_obj,
        incumbent_sol,
        tts,
    }
}
