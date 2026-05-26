//! Diverse solution-pool hybrid heuristic for HUBO problems.
//!
//! Maintains a pool of solutions and applies multiple variation operators:
//! random one-flip, XOR-style recombination with a peer, and best-improving
//! single flip. Tracks per-operator benefit statistics.

use std::time::Instant;

use crate::Logger;
use crate::coeff::Coeff;
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Rng, Status, base_seed, random_solution};

/// Pool-hybrid-specific configuration.
pub struct Config {
    /// Parameters shared with all heuristics.
    pub common: CommonConfig,
    /// Maximum number of pool members retained.
    pub pool_size: usize,
    /// Number of initial random solutions.
    pub init_solutions: usize,
    /// Maximum offspring iterations (None = unbounded).
    pub max_iterations: Option<u64>,
    /// Maximum number of XOR-applied differing bits per move.
    pub xor_max_flips: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            pool_size: 128,
            init_solutions: 32,
            max_iterations: None,
            xor_max_flips: 8,
        }
    }
}

#[derive(Clone)]
struct Candidate<C: Coeff> {
    sol: BitSolution,
    obj: C,
}

#[derive(Clone, Copy)]
enum Operator {
    OneFlip,
    Xor,
    BestFlip,
}

impl Operator {
    fn all() -> [Self; 3] {
        [Self::OneFlip, Self::Xor, Self::BestFlip]
    }

    fn idx(self) -> usize {
        match self {
            Self::OneFlip => 0,
            Self::Xor => 1,
            Self::BestFlip => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::OneFlip => "one_flip",
            Self::Xor => "xor",
            Self::BestFlip => "best_flip",
        }
    }
}

#[derive(Clone, Copy, Default)]
struct OpStats {
    attempts: u64,
    benefited: u64,
    improved_incumbent: u64,
}

/// Run the pool-hybrid heuristic on a HUBO instance.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &Config,
    logger: &Logger,
) -> HeuristicResult<C> {
    let _ = logger;
    let start = Instant::now();
    let n = instance.n_vars();

    let pool_cap = config.pool_size.max(1);
    let init_solutions = config.init_solutions.max(pool_cap);
    let xor_max_flips = config.xor_max_flips.max(1);

    log::info!(
        "Pool: n_vars={}, n_terms={}, pool_size={}, init_solutions={}, xor_max_flips={}, max_iter={}",
        n,
        instance.n_terms(),
        pool_cap,
        init_solutions,
        xor_max_flips,
        config
            .max_iterations
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unbounded".to_string())
    );

    let seed = base_seed(config.common.seed);
    let mut rng = Rng::new(seed);

    let mut pool: Vec<Candidate<C>> = Vec::with_capacity(pool_cap);
    let mut best_obj = C::max_value();
    let mut best_sol: Option<BitSolution> = None;
    let mut best_tts = 0.0f64;

    let mut timed_out = false;
    let mut cutoff_reached = false;
    let mut interrupted = false;
    let mut total_iters = 0u64;

    for _ in 0..init_solutions {
        if let Some(tl) = config.common.time_limit
            && start.elapsed().as_secs_f64() >= tl
        {
            timed_out = true;
            break;
        }

        let sol = random_solution(n, instance.var_type(), &mut rng);
        let obj = sol.evaluate(instance);

        if obj < best_obj {
            best_obj = obj;
            best_sol = Some(sol.clone());
            best_tts = start.elapsed().as_secs_f64();
            log::debug!(
                "new incumbent: obj = {:.6}, tts = {:.3}s",
                best_obj,
                best_tts
            );

            if let Some(cutoff) = config.common.cutoff
                && best_obj.to_f64() <= cutoff
            {
                cutoff_reached = true;
            }
        }

        try_insert_into_pool(&mut pool, pool_cap, Candidate { sol, obj }, n);
    }

    if pool.is_empty() {
        // Defensive fallback for degenerate limits.
        let sol = random_solution(n, instance.var_type(), &mut rng);
        let obj = sol.evaluate(instance);
        pool.push(Candidate {
            sol: sol.clone(),
            obj,
        });
        best_obj = obj;
        best_sol = Some(sol);
    }

    let mut stats = [OpStats::default(); 3];
    let ops = Operator::all();

    while !timed_out && !cutoff_reached && !interrupted {
        if crate::interrupt::is_interrupted() {
            interrupted = true;
            break;
        }

        if let Some(limit) = config.max_iterations
            && total_iters >= limit
        {
            break;
        }

        if let Some(tl) = config.common.time_limit
            && start.elapsed().as_secs_f64() >= tl
        {
            timed_out = true;
            break;
        }

        let base_idx = rng.index(pool.len());
        let base = pool[base_idx].clone();

        let mut child_sol = base.sol.clone();
        let mut child_obj = base.obj;

        let mut term_state = instance.init_term_state(&child_sol);
        let mut delta_cache = instance.init_delta_cache(&child_sol, &term_state);

        let op = ops[(total_iters as usize) % ops.len()];
        stats[op.idx()].attempts += 1;

        match op {
            Operator::OneFlip => {
                if n > 0 {
                    let var = rng.index(n);
                    let delta = delta_cache.deltas[var];
                    instance.flip_with_term_state(var, &mut child_sol, &mut term_state);
                    child_obj += delta;
                    instance.update_delta_cache_after_flip(
                        var,
                        &child_sol,
                        &term_state,
                        &mut delta_cache,
                    );
                }
            }
            Operator::BestFlip => {
                if n > 0 {
                    let mut best_var = 0usize;
                    let mut best_delta = delta_cache.deltas[0];
                    for var in 1..n {
                        let delta = delta_cache.deltas[var];
                        if delta < best_delta {
                            best_delta = delta;
                            best_var = var;
                        }
                    }
                    // If no improving move exists, still perturb with one random flip.
                    let var = if best_delta < C::zero() {
                        best_var
                    } else {
                        rng.index(n)
                    };
                    let delta = delta_cache.deltas[var];
                    instance.flip_with_term_state(var, &mut child_sol, &mut term_state);
                    child_obj += delta;
                    instance.update_delta_cache_after_flip(
                        var,
                        &child_sol,
                        &term_state,
                        &mut delta_cache,
                    );
                }
            }
            Operator::Xor => {
                apply_xor_move(
                    instance,
                    &pool,
                    &mut rng,
                    base_idx,
                    xor_max_flips,
                    &mut child_sol,
                    &mut child_obj,
                    &mut term_state,
                    &mut delta_cache,
                );
            }
        }

        if child_obj < base.obj {
            stats[op.idx()].benefited += 1;
        }

        if child_obj < best_obj {
            best_obj = child_obj;
            best_sol = Some(child_sol.clone());
            best_tts = start.elapsed().as_secs_f64();
            stats[op.idx()].improved_incumbent += 1;
            log::debug!(
                "new incumbent: obj = {:.6}, tts = {:.3}s",
                best_obj,
                best_tts
            );

            if let Some(cutoff) = config.common.cutoff
                && best_obj.to_f64() <= cutoff
            {
                cutoff_reached = true;
            }
        }

        try_insert_into_pool(
            &mut pool,
            pool_cap,
            Candidate {
                sol: child_sol,
                obj: child_obj,
            },
            n,
        );

        total_iters += 1;
    }

    for op in ops {
        let st = stats[op.idx()];
        let rate = if st.attempts == 0 {
            0.0
        } else {
            st.benefited as f64 / st.attempts as f64
        };
        log::info!(
            "Pool op={} attempts={} benefited={} improve_rate={:.3} incumbent_improvements={}",
            op.name(),
            st.attempts,
            st.benefited,
            rate,
            st.improved_incumbent
        );
    }

    log::info!(
        "Pool diversity: avg_pairwise_hamming={:.3}",
        avg_pairwise_hamming(&pool, n)
    );

    let solving_time = start.elapsed().as_secs_f64();
    let status = if interrupted {
        Status::Interrupted
    } else if cutoff_reached {
        Status::Cutoff
    } else if timed_out {
        Status::TimeLimit
    } else {
        Status::Completed
    };

    let solution = best_sol.expect("Pool heuristic always has at least one solution");
    let objective = solution.evaluate(instance);

    let result = HeuristicResult {
        method: "Pool",
        status,
        objective,
        solution,
        solving_time,
        tts: best_tts,
        iterations: total_iters,
    };

    if let Some(ref path) = config.common.solution_file {
        match result.write_solution_file(path, instance.var_type()) {
            Ok(()) => log::info!("solution written to {}", path),
            Err(e) => log::error!("failed to write solution: {}", e),
        }
    }

    result
}

fn apply_xor_move<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    pool: &[Candidate<C>],
    rng: &mut Rng,
    base_idx: usize,
    xor_max_flips: usize,
    child_sol: &mut BitSolution,
    child_obj: &mut C,
    term_state: &mut crate::state::TermState<V>,
    delta_cache: &mut crate::state::DeltaCache<C>,
) {
    let n = instance.n_vars();
    if n == 0 {
        return;
    }

    if pool.len() < 2 {
        let var = rng.index(n);
        let delta = delta_cache.deltas[var];
        instance.flip_with_term_state(var, child_sol, term_state);
        *child_obj += delta;
        instance.update_delta_cache_after_flip(var, child_sol, term_state, delta_cache);
        return;
    }

    let mut peer_idx = rng.index(pool.len() - 1);
    if peer_idx >= base_idx {
        peer_idx += 1;
    }
    let peer = &pool[peer_idx].sol;

    let mut differing: Vec<usize> = Vec::new();
    differing.reserve(n);
    for i in 0..n {
        if child_sol.values.contains(i) != peer.values.contains(i) {
            differing.push(i);
        }
    }

    if differing.is_empty() {
        let var = rng.index(n);
        let delta = delta_cache.deltas[var];
        instance.flip_with_term_state(var, child_sol, term_state);
        *child_obj += delta;
        instance.update_delta_cache_after_flip(var, child_sol, term_state, delta_cache);
        return;
    }

    let max_flips = xor_max_flips.min(differing.len());
    let flips = 1 + rng.index(max_flips);

    for _ in 0..flips {
        let p = rng.index(differing.len());
        let var = differing.swap_remove(p);
        let delta = delta_cache.deltas[var];
        instance.flip_with_term_state(var, child_sol, term_state);
        *child_obj += delta;
        instance.update_delta_cache_after_flip(var, child_sol, term_state, delta_cache);

        if differing.is_empty() {
            break;
        }
    }
}

fn try_insert_into_pool<C: Coeff>(
    pool: &mut Vec<Candidate<C>>,
    cap: usize,
    candidate: Candidate<C>,
    n: usize,
) {
    if let Some(pos) = pool
        .iter()
        .position(|c| hamming_distance(&c.sol, &candidate.sol, n) == 0)
    {
        if candidate.obj < pool[pos].obj {
            pool[pos] = candidate;
        }
        return;
    }

    if pool.len() < cap {
        pool.push(candidate);
        return;
    }

    let mut elite_idx = 0usize;
    for i in 1..pool.len() {
        if pool[i].obj < pool[elite_idx].obj {
            elite_idx = i;
        }
    }

    let cand_min_dist = pool
        .iter()
        .map(|c| hamming_distance(&c.sol, &candidate.sol, n))
        .min()
        .unwrap_or(n);

    let mut replace_idx: Option<usize> = None;
    let mut worst_crowding = usize::MAX;

    for i in 0..pool.len() {
        if i == elite_idx {
            continue;
        }
        let crowding = min_dist_to_others(pool, i, n);
        if crowding < worst_crowding {
            worst_crowding = crowding;
            replace_idx = Some(i);
        } else if crowding == worst_crowding
            && let Some(j) = replace_idx
            && pool[i].obj > pool[j].obj
        {
            replace_idx = Some(i);
        }
    }

    if let Some(i) = replace_idx {
        let victim_crowding = min_dist_to_others(pool, i, n);
        if cand_min_dist > victim_crowding
            || (cand_min_dist == victim_crowding && candidate.obj < pool[i].obj)
        {
            pool[i] = candidate;
        }
    }
}

fn min_dist_to_others<C: Coeff>(pool: &[Candidate<C>], idx: usize, n: usize) -> usize {
    let mut best = usize::MAX;
    for (j, c) in pool.iter().enumerate() {
        if j == idx {
            continue;
        }
        let d = hamming_distance(&pool[idx].sol, &c.sol, n);
        if d < best {
            best = d;
        }
    }
    if best == usize::MAX { n } else { best }
}

fn avg_pairwise_hamming<C: Coeff>(pool: &[Candidate<C>], n: usize) -> f64 {
    if pool.len() < 2 {
        return 0.0;
    }

    let mut sum = 0usize;
    let mut pairs = 0usize;
    for i in 0..pool.len() {
        for j in (i + 1)..pool.len() {
            sum += hamming_distance(&pool[i].sol, &pool[j].sol, n);
            pairs += 1;
        }
    }

    sum as f64 / pairs as f64
}

fn hamming_distance(a: &BitSolution, b: &BitSolution, n: usize) -> usize {
    let mut dist = 0usize;
    for i in 0..n {
        if a.values.contains(i) != b.values.contains(i) {
            dist += 1;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Status;
    use crate::model::HuboModel;

    #[test]
    fn finds_optimum_bin() {
        let instance = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_linear(2, 1.0)
            .build();

        let config = Config {
            common: CommonConfig {
                seed: Some(42),
                ..Default::default()
            },
            max_iterations: Some(500),
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert!((result.objective).abs() < 1e-10);
    }

    #[test]
    fn finds_optimum_spin() {
        let instance = HuboModel::spin(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_linear(2, 1.0)
            .build();

        let config = Config {
            common: CommonConfig {
                seed: Some(42),
                ..Default::default()
            },
            max_iterations: Some(500),
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert!((result.objective - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn cutoff() {
        let mut builder = HuboModel::binary(60);
        for i in 0..60 {
            builder = builder.add_linear(i, 1.0);
        }
        let instance = builder.build();

        let config = Config {
            common: CommonConfig {
                cutoff: Some(60.0),
                seed: Some(7),
                ..Default::default()
            },
            max_iterations: Some(1000),
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert_eq!(result.status, Status::Cutoff);
    }
}
