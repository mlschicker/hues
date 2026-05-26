//! Parallel Tempering heuristic for HUSO/HUBO optimization.
//!
//! Implements the PT+ feature set from arXiv:2603.13607:
//!   1. DEO (non-reversible) swap scheme  — O(R) round-trip vs O(R²) for random swaps
//!   2. Adaptive temperature ladder        — auto-calibrates to target swap acceptance rate
//!   3. Greedy descent after accepted swaps — drives new cold configurations to local minima
//!   4. Parallel independent runs          — embarrassingly parallel via rayon

use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;

use crate::coeff::Coeff;
use crate::state::{DeltaCache, TermState};
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Status, base_seed};

// ============================================================
//  Mutable state of one replica
// ============================================================

struct State<C: Coeff, V: VarDomain> {
    sigma: BitSolution,
    term_state: TermState<V>,
    delta_cache: DeltaCache<C>,
    energy: C,
}

impl<C: Coeff, V: VarDomain> State<C, V> {
    fn from_sigma(instance: &HuboInstance<C, V>, solution: &BitSolution) -> Self {
        let term_state = instance.init_term_state(solution);
        let delta_cache = instance.init_delta_cache(solution, &term_state);
        let energy = solution.evaluate(instance);
        State {
            sigma: solution.clone(),
            term_state,
            delta_cache,
            energy,
        }
    }

    fn random(instance: &HuboInstance<C, V>, rng: &mut impl Rng) -> Self {
        let sigma = BitSolution::from_bool_vec(
            (0..instance.n_vars())
                .map(|_| rng.random_bool(0.5))
                .collect(),
        );
        Self::from_sigma(instance, &sigma)
    }

    #[inline(always)]
    fn delta_e(&self, j: usize) -> C {
        self.delta_cache.deltas[j]
    }

    #[inline]
    fn flip(&mut self, j: usize, instance: &HuboInstance<C, V>) {
        self.energy += self.delta_cache.deltas[j];
        instance.flip_with_term_state(j, &mut self.sigma, &mut self.term_state);
        instance.update_delta_cache_after_flip(
            j,
            &self.sigma,
            &self.term_state,
            &mut self.delta_cache,
        );
    }
}

// ============================================================
//  Move functions
// ============================================================

/// One Metropolis sweep: visit all n variables in uniformly random order,
/// accept flip j with probability min(1, exp(-β · ΔE_j)).
fn metropolis_sweep<C: Coeff, V: VarDomain>(
    state: &mut State<C, V>,
    instance: &HuboInstance<C, V>,
    beta: f64,
    rng: &mut impl Rng,
    order: &mut [usize],
) {
    let n = instance.n_vars();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
    for &j in order.iter() {
        let de = state.delta_e(j).to_f64();
        if de <= 0.0 || rng.random::<f64>() < (-beta * de).exp() {
            state.flip(j, instance);
        }
    }
}

/// Greedy descent toward a local minimum: scan all variables and accept
/// every improving flip.  Capped at `max_passes` full sweeps so big
/// instances don't spend disproportionate time here.
fn greedy_descent<C: Coeff, V: VarDomain>(
    state: &mut State<C, V>,
    instance: &HuboInstance<C, V>,
    max_passes: usize,
) {
    for _ in 0..max_passes {
        let mut improved = false;
        for j in 0..instance.n_vars() {
            if state.delta_e(j) < C::zero() {
                state.flip(j, instance);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
}

// ============================================================
//  Temperature ladder
// ============================================================

struct Ladder {
    /// betas[0] = 1/T_min  (coldest replica, highest β).
    /// betas[R-1] = 1/T_max (hottest replica, lowest β).
    betas: Vec<f64>,
    /// Per-pair swap statistics: [accepted, proposed].
    swap_stats: Vec<[u64; 2]>,
}

impl Ladder {
    /// Geometric spacing: T_i = T_min · (T_max/T_min)^(i/(R-1)).
    fn geometric(r: usize, t_min: f64, t_max: f64) -> Self {
        assert!(r >= 2, "need at least 2 replicas");
        assert!(t_min > 0.0 && t_min < t_max, "need 0 < t_min < t_max");
        let betas = (0..r)
            .map(|i| {
                let t = t_min * (t_max / t_min).powf(i as f64 / (r - 1) as f64);
                1.0 / t
            })
            .collect::<Vec<_>>();
        Ladder {
            swap_stats: vec![[0u64; 2]; r - 1],
            betas,
        }
    }

    fn n_replicas(&self) -> usize {
        self.betas.len()
    }

    fn record(&mut self, ri: usize, accepted: bool) {
        self.swap_stats[ri][1] += 1;
        if accepted {
            self.swap_stats[ri][0] += 1;
        }
    }

    fn accept_rate(&self, ri: usize) -> f64 {
        let [a, n] = self.swap_stats[ri];
        if n == 0 { 0.5 } else { a as f64 / n as f64 }
    }

    /// Adapt interior temperatures toward `target` swap acceptance rate.
    fn adapt(&mut self, target: f64) {
        let r = self.betas.len();
        let mut temps: Vec<f64> = self.betas.iter().map(|&b| 1.0 / b).collect();

        for ri in 0..r - 1 {
            if ri + 1 >= r - 1 {
                continue;
            }
            let rate = self.accept_rate(ri);
            let log_ratio = (rate / target).ln().clamp(-0.2, 0.2);
            let t_lo = temps[ri];
            let t_hi = temps[ri + 2];
            let t_cur = temps[ri + 1];
            let frac = (t_cur - t_lo) / (t_hi - t_lo);
            let new_frac = (frac * (1.0 + log_ratio)).clamp(0.001, 0.999);
            temps[ri + 1] = t_lo + new_frac * (t_hi - t_lo);
        }

        for (i, b) in self.betas.iter_mut().enumerate() {
            *b = 1.0 / temps[i];
        }
        for s in &mut self.swap_stats {
            *s = [0, 0];
        }
    }
}

// ============================================================
//  Configuration and result
// ============================================================

/// Configuration for the standalone parallel tempering entry point.
#[derive(Clone, Debug)]
pub struct PtConfig {
    pub n_replicas: usize,
    pub n_runs: usize,
    pub n_sweeps: usize,
    pub swap_interval: usize,
    pub t_min: f64,
    pub t_max: f64,
    pub greedy_after_swap: bool,
    /// Maximum greedy-descent passes per invocation.  Caps `O(n²)` blowup on
    /// big instances; small enough that PT stays responsive even at scale.
    pub greedy_max_passes: usize,
    pub adapt_interval: usize,
    pub target_accept_rate: f64,
    /// Optional warm-start for the cold replica of every run.
    pub warm_start: Option<BitSolution>,
    /// Wall-clock seconds; runs abort after each sweep when exceeded.
    pub time_limit: Option<f64>,
    /// Stop as soon as a replica's energy reaches `cutoff`.
    pub cutoff: Option<f64>,
}

impl Default for PtConfig {
    fn default() -> Self {
        PtConfig {
            n_replicas: 12,
            n_runs: 8,
            n_sweeps: 10_000,
            swap_interval: 5,
            t_min: 0.1,
            t_max: 10.0,
            greedy_after_swap: true,
            greedy_max_passes: 4,
            adapt_interval: 500,
            target_accept_rate: 0.25,
            warm_start: None,
            time_limit: None,
            cutoff: None,
        }
    }
}

/// Parallel tempering configuration for the main heuristic interface.
pub struct Config {
    pub common: CommonConfig,
    pub n_replicas: usize,
    pub n_runs: usize,
    pub n_sweeps: usize,
    pub swap_interval: usize,
    pub t_min: f64,
    pub t_max: f64,
    pub greedy_after_swap: bool,
    pub adapt_interval: usize,
    pub target_accept_rate: f64,
}

impl Default for Config {
    fn default() -> Self {
        let pt = PtConfig::default();
        Self {
            common: CommonConfig::default(),
            n_replicas: pt.n_replicas,
            n_runs: pt.n_runs,
            n_sweeps: pt.n_sweeps,
            swap_interval: pt.swap_interval,
            t_min: pt.t_min,
            t_max: pt.t_max,
            greedy_after_swap: pt.greedy_after_swap,
            adapt_interval: pt.adapt_interval,
            target_accept_rate: pt.target_accept_rate,
        }
    }
}

/// Output of a parallel tempering run.
#[derive(Clone)]
pub struct PtResult<C: Coeff> {
    pub best_energy: C,
    pub best_solution: BitSolution,
    pub swap_accept_rate: f64,
}

// ============================================================
//  Core PT runner  (one independent ladder)
// ============================================================

fn run_single<C: Coeff, V: VarDomain>(
    inst: &HuboInstance<C, V>,
    config: &PtConfig,
    seed: u64,
    run_id: u64,
    start: Instant,
) -> PtResult<C> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut ladder = Ladder::geometric(config.n_replicas, config.t_min, config.t_max);
    let r = ladder.n_replicas();

    let mut replicas: Vec<State<C, V>> = (0..r)
        .map(|ri| {
            if ri == 0 {
                if let Some(ref ws) = config.warm_start {
                    State::from_sigma(inst, ws)
                } else {
                    State::random(inst, &mut rng)
                }
            } else {
                State::random(inst, &mut rng)
            }
        })
        .collect();

    greedy_descent(&mut replicas[0], inst, config.greedy_max_passes);

    let mut best_energy = replicas[0].energy;
    let mut best_solution = replicas[0].sigma.clone();
    log::debug!("run {}: initial obj = {:.6}", run_id, best_energy.to_f64());

    let mut order: Vec<usize> = (0..inst.n_vars()).collect();

    let mut total_accepted = 0u64;
    let mut total_proposed = 0u64;

    let mut deo_parity = 0usize;

    let should_stop = |best: C| -> bool {
        if crate::interrupt::is_interrupted() {
            return true;
        }
        if let Some(tl) = config.time_limit
            && start.elapsed().as_secs_f64() >= tl
        {
            return true;
        }
        if let Some(co) = config.cutoff
            && best.to_f64() <= co
        {
            return true;
        }
        false
    };

    for sweep in 0..config.n_sweeps {
        if should_stop(best_energy) {
            break;
        }

        for (ri, replica) in replicas.iter_mut().enumerate() {
            metropolis_sweep(replica, inst, ladder.betas[ri], &mut rng, &mut order);
        }

        if replicas[0].energy < best_energy {
            best_energy = replicas[0].energy;
            best_solution = replicas[0].sigma.clone();
            log::debug!(
                "new incumbent: obj = {:.6}, tts = {:.3}s",
                best_energy.to_f64(),
                start.elapsed().as_secs_f64()
            );
        }

        if (sweep + 1) % config.swap_interval == 0 {
            let parity_start = deo_parity;
            deo_parity = 1 - deo_parity;

            let mut ri = parity_start;
            while ri + 1 < r {
                total_proposed += 1;

                // log P_acc = (β_ri - β_{ri+1}) · (E_ri - E_{ri+1})
                let e_diff = (replicas[ri].energy - replicas[ri + 1].energy).to_f64();
                let log_acc = (ladder.betas[ri] - ladder.betas[ri + 1]) * e_diff;

                let accepted = log_acc >= 0.0 || rng.random::<f64>() < log_acc.exp();
                ladder.record(ri, accepted);

                if accepted {
                    replicas.swap(ri, ri + 1);
                    total_accepted += 1;

                    if config.greedy_after_swap {
                        greedy_descent(&mut replicas[ri], inst, config.greedy_max_passes);
                    }

                    if replicas[ri].energy < best_energy {
                        best_energy = replicas[ri].energy;
                        best_solution = replicas[ri].sigma.clone();
                        log::debug!(
                            "new incumbent: obj = {:.6}, tts = {:.3}s",
                            best_energy.to_f64(),
                            start.elapsed().as_secs_f64()
                        );
                    }
                }

                ri += 2;
            }
        }

        if config.adapt_interval > 0 && (sweep + 1) % config.adapt_interval == 0 {
            ladder.adapt(config.target_accept_rate);
        }
    }

    let swap_accept_rate = if total_proposed > 0 {
        total_accepted as f64 / total_proposed as f64
    } else {
        0.0
    };

    log::debug!(
        "run {} done: best={:.6}, swap_rate={:.3}",
        run_id,
        best_energy.to_f64(),
        swap_accept_rate
    );

    PtResult {
        best_energy,
        best_solution,
        swap_accept_rate,
    }
}

// ============================================================
//  Public entry point
// ============================================================

/// Run parallel tempering with `config.n_runs` independent ladders in parallel.
pub fn parallel_tempering<C: Coeff, V: VarDomain>(
    inst: &HuboInstance<C, V>,
    config: &PtConfig,
) -> PtResult<C> {
    parallel_tempering_with_seed(inst, config, 0)
}

/// Same as [`parallel_tempering`] but allows controlling the base RNG seed.
pub fn parallel_tempering_with_seed<C: Coeff, V: VarDomain>(
    inst: &HuboInstance<C, V>,
    config: &PtConfig,
    base_seed: u64,
) -> PtResult<C> {
    let start = Instant::now();
    let results: Vec<PtResult<C>> = (0u64..config.n_runs as u64)
        .into_par_iter()
        .map(|id| {
            run_single(
                inst,
                config,
                base_seed.wrapping_add(id.wrapping_mul(6364136223846793005).wrapping_add(1)),
                id,
                start,
            )
        })
        .collect();

    let best = results
        .iter()
        .min_by(|a, b| a.best_energy.partial_cmp(&b.best_energy).unwrap())
        .unwrap();

    let avg_rate = results.iter().map(|r| r.swap_accept_rate).sum::<f64>() / results.len() as f64;

    PtResult {
        best_energy: best.best_energy,
        best_solution: best.best_solution.clone(),
        swap_accept_rate: avg_rate,
    }
}

// ============================================================
//  Temperature calibration helpers
// ============================================================

/// Estimate a reasonable T_min: the temperature where acceptance first drops below `target_rate`.
pub fn calibrate_t_min<C: Coeff, V: VarDomain>(
    inst: &HuboInstance<C, V>,
    target_rate: f64,
    seed: u64,
) -> f64 {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut state = State::random(inst, &mut rng);
    greedy_descent(&mut state, inst, 4);
    let mut order: Vec<usize> = (0..inst.n_vars()).collect();

    let mut t = 10.0f64;
    loop {
        let beta = 1.0 / t;
        let mut accepted = 0u64;
        let mut proposed = 0u64;
        for _ in 0..10 {
            let n = inst.n_vars();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                order.swap(i, j);
            }
            for &j in order.iter() {
                proposed += 1;
                let de = state.delta_e(j).to_f64();
                if de <= 0.0 || rng.random::<f64>() < (-beta * de).exp() {
                    state.flip(j, inst);
                    accepted += 1;
                }
            }
        }
        let rate = accepted as f64 / proposed as f64;
        if rate < target_rate {
            return t * 1.5;
        }
        t *= 0.8;
        if t < 1e-6 {
            return 1e-4;
        }
    }
}

/// Estimate a reasonable T_max: the temperature at which ~90% of uphill moves are accepted.
pub fn calibrate_t_max<C: Coeff, V: VarDomain>(inst: &HuboInstance<C, V>, seed: u64) -> f64 {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut state = State::random(inst, &mut rng);
    let mut order: Vec<usize> = (0..inst.n_vars()).collect();

    let mut t = 0.1f64;
    loop {
        let beta = 1.0 / t;
        let mut acc = 0u64;
        let mut prop = 0u64;
        let n = inst.n_vars();
        for _ in 0..5 {
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                order.swap(i, j);
            }
            for &j in order.iter() {
                prop += 1;
                let de = state.delta_e(j).to_f64();
                if de <= 0.0 || rng.random::<f64>() < (-beta * de).exp() {
                    state.flip(j, inst);
                    acc += 1;
                }
            }
        }
        if acc as f64 / prop as f64 > 0.90 {
            return t;
        }
        t *= 1.5;
        if t > 1e6 {
            return 1e4;
        }
    }
}

// ============================================================
//  Heuristic interface
// ============================================================

/// Run parallel tempering through the common heuristic interface used by the CLI.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &Config,
    initial_solution: Option<Vec<C>>,
) -> HeuristicResult<C> {
    let start = Instant::now();

    let warm_start = initial_solution
        .as_ref()
        .map(|vals| BitSolution::from_vec(vals));

    let defaults = PtConfig::default();
    let pt_config = PtConfig {
        n_replicas: config.n_replicas,
        n_runs: config.n_runs,
        n_sweeps: config.n_sweeps,
        swap_interval: config.swap_interval,
        t_min: config.t_min,
        t_max: config.t_max,
        greedy_after_swap: config.greedy_after_swap,
        greedy_max_passes: defaults.greedy_max_passes,
        adapt_interval: config.adapt_interval,
        target_accept_rate: config.target_accept_rate,
        warm_start,
        time_limit: config.common.time_limit,
        cutoff: config.common.cutoff,
    };

    log::info!(
        "PT: n_vars={}, n_terms={}, n_runs={}, n_replicas={}, n_sweeps={}, T_min={}, T_max={}",
        instance.n_vars(),
        instance.n_terms(),
        config.n_runs,
        config.n_replicas,
        config.n_sweeps,
        config.t_min,
        config.t_max
    );

    let seed = base_seed(config.common.seed);
    let pt_result = parallel_tempering_with_seed(instance, &pt_config, seed);

    let solution = pt_result.best_solution;
    let objective = solution.evaluate(instance);
    let solving_time = start.elapsed().as_secs_f64();

    let status = if crate::interrupt::is_interrupted() {
        Status::Interrupted
    } else if config
        .common
        .cutoff
        .is_some_and(|cutoff| objective.to_f64() <= cutoff)
    {
        Status::Cutoff
    } else if config
        .common
        .time_limit
        .is_some_and(|tl| solving_time >= tl)
    {
        Status::TimeLimit
    } else {
        Status::Completed
    };

    let iterations = (config.n_runs as u64)
        .saturating_mul(config.n_replicas as u64)
        .saturating_mul(config.n_sweeps as u64)
        .saturating_mul(instance.n_vars() as u64);

    log::info!(
        "PT finished: status={}, obj={}, iters={}, time={:.3}s, swap_rate={:.3}",
        status,
        objective,
        iterations,
        solving_time,
        pt_result.swap_accept_rate
    );

    let result = HeuristicResult {
        method: "PT",
        status,
        objective,
        solution,
        solving_time,
        tts: solving_time,
        iterations,
    };

    if let Some(ref path) = config.common.solution_file {
        match result.write_solution_file(path, instance.var_type()) {
            Ok(()) => log::info!("solution written to {}", path),
            Err(e) => log::error!("failed to write solution: {}", e),
        }
    }

    result
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HuboModel;

    fn frustrated_triangle(j: f64) -> HuboInstance<f64, crate::domain::Spin> {
        HuboModel::spin(3)
            .add_quadratic(0, 1, j)
            .add_quadratic(1, 2, j)
            .add_quadratic(0, 2, j)
            .build()
    }

    #[test]
    fn test_energy_and_fields() {
        let inst = frustrated_triangle(1.0);
        // sigma = [+1, +1, -1]
        let sigma = BitSolution::from_vec(&[1.0f64, 1.0, -1.0]);
        let energy = sigma.evaluate(&inst);
        assert!((energy - (-1.0)).abs() < 1e-12);

        let h = sigma.local_fields(&inst);
        // h[0] = s1 + s2 = 1 + (-1) = 0
        assert!((h[0] - 0.0).abs() < 1e-12);
        // h[1] = s0 + s2 = 1 + (-1) = 0
        assert!((h[1] - 0.0).abs() < 1e-12);
        // h[2] = s1 + s0 = 1 + 1 = 2
        assert!((h[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_flip_incremental_vs_full() {
        let inst = frustrated_triangle(2.5);
        let sigma = BitSolution::from_vec(&[1.0f64, -1.0, 1.0]);
        let mut state = State::from_sigma(&inst, &sigma);

        state.flip(1, &inst);

        let full_e = state.sigma.evaluate(&inst);
        assert!(
            (state.energy - full_e).abs() < 1e-10,
            "incremental energy {} != full {}",
            state.energy,
            full_e
        );

        let full_ts = inst.init_term_state(&state.sigma);
        let full_dc = inst.init_delta_cache(&state.sigma, &full_ts);
        for j in 0..inst.n_vars() {
            assert!(
                (state.delta_cache.deltas[j] - full_dc.deltas[j]).abs() < 1e-10,
                "delta[{}] mismatch: incremental={} full={}",
                j,
                state.delta_cache.deltas[j],
                full_dc.deltas[j]
            );
        }
    }

    #[test]
    fn test_greedy_descent_reaches_local_min() {
        let inst = frustrated_triangle(1.0);
        // All aligned: maximum energy state.
        let sigma = BitSolution::from_vec(&[1.0f64, 1.0, 1.0]);
        let mut state = State::from_sigma(&inst, &sigma);

        greedy_descent(&mut state, &inst, 8);

        for j in 0..inst.n_vars() {
            assert!(
                state.delta_e(j) >= 0.0,
                "variable {} is still improvable after greedy descent",
                j
            );
        }
    }

    #[test]
    fn test_pt_finds_ground_state() {
        let inst = frustrated_triangle(1.0);
        let cfg = PtConfig {
            n_replicas: 8,
            n_runs: 4,
            n_sweeps: 2_000,
            t_min: 0.05,
            t_max: 5.0,
            ..Default::default()
        };
        let res = parallel_tempering(&inst, &cfg);
        assert!(
            res.best_energy <= -1.0 + 1e-9,
            "PT failed to find ground state: got {}",
            res.best_energy
        );
    }

    #[test]
    fn test_swap_acceptance_formula() {
        let beta_cold = 2.0f64;
        let beta_hot = 0.5f64;
        let e_cold = 3.0f64;
        let e_hot = 1.0f64;
        let log_acc = (beta_cold - beta_hot) * (e_cold - e_hot);
        assert!(log_acc > 0.0, "should always accept when cold is worse");
    }
}
