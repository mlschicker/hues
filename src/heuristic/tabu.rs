//! Tabu Search heuristic for HUBO problems.
//!
//! Uses a short-term memory (tabu list) to prevent cycling: recently
//! flipped variables are forbidden for a configurable number of
//! iterations unless the flip leads to a new global best (aspiration).

use std::time::Instant;

use crate::coeff::Coeff;
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Rng, Status, base_seed, random_solution};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tabu-search-specific configuration.
pub struct Config {
    /// Parameters shared with all heuristics.
    pub common: CommonConfig,
    /// Tabu tenure — number of iterations a variable stays tabu after
    /// being flipped.  `None` → `sqrt(n_vars)` (a common default).
    pub tenure: Option<usize>,
    /// Maximum iterations per restart (each iteration evaluates all
    /// `n_vars` neighbours).  `None` → no iteration cap
    /// (solver stops only on time limit, cutoff, or restart exhaustion).
    pub max_iterations: Option<u64>,
    /// Number of independent restarts.
    pub restarts: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            tenure: None,
            max_iterations: None,
            restarts: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Run tabu search on a HUBO instance.
///
/// `initial_solution`, when provided, is used as the starting point for
/// restart 0; subsequent restarts use random initial assignments to keep
/// diversification.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &Config,
    initial_solution: Option<&BitSolution>,
) -> HeuristicResult<C> {
    let start = Instant::now();
    let n = instance.n_vars();

    let tenure = config
        .tenure
        .unwrap_or_else(|| (n as f64).sqrt().ceil() as usize);
    let max_iter = config.max_iterations;

    let max_iter_str = match max_iter {
        Some(v) => v.to_string(),
        None => "unbounded".to_string(),
    };

    log::info!(
        "Tabu: n_vars={}, n_terms={}, restarts={}, tenure={}, max_iter={}",
        n,
        instance.n_terms(),
        config.restarts,
        tenure,
        max_iter_str
    );

    let seed = base_seed(config.common.seed);

    let mut best_obj = C::max_value();
    let mut best_sol: Option<BitSolution> = None;
    let mut best_tts: f64 = 0.0;
    let mut total_iters: u64 = 0;
    let mut timed_out = false;
    let mut cutoff_reached = false;
    let mut interrupted = false;

    for restart in 0..config.restarts {
        let mut rng = Rng::new(seed.wrapping_add(restart as u64));

        let mut sol: BitSolution = match (restart, initial_solution) {
            (0, Some(init)) if init.values.len() == n => init.clone(),
            _ => random_solution(n, instance.var_type(), &mut rng),
        };

        let mut term_state = instance.init_term_state(&sol);
        let mut delta_cache = instance.init_delta_cache(&sol, &term_state);

        let mut obj = sol.evaluate(instance);
        if obj < best_obj {
            best_obj = obj;
            best_sol = Some(sol.clone());
            best_tts = start.elapsed().as_secs_f64();
            log::debug!(
                "new incumbent: obj = {:.6}, tts = {:.3}s",
                best_obj,
                best_tts
            );
        }

        log::debug!("restart {}: initial obj = {}", restart, obj);

        // tabu_until[i] = iteration number until which variable i is tabu.
        let mut tabu_until = vec![0u64; n];
        let mut restart_iters: u64 = 0;

        let mut iter: u64 = 0;
        loop {
            if crate::interrupt::is_interrupted() {
                interrupted = true;
                break;
            }

            if let Some(limit) = max_iter
                && iter >= limit
            {
                break;
            }

            if let Some(tl) = config.common.time_limit
                && start.elapsed().as_secs_f64() >= tl
            {
                timed_out = true;
                break;
            }

            // Evaluate all neighbours and pick the best admissible move.
            let mut best_var: Option<usize> = None;
            let mut best_delta = C::max_value();

            for (var, item) in tabu_until.iter().enumerate().take(n) {
                let delta = delta_cache.deltas[var];

                let is_tabu = *item > iter;
                let aspiration = obj + delta < best_obj;

                if (!is_tabu || aspiration) && delta < best_delta {
                    best_delta = delta;
                    best_var = Some(var);
                }
            }

            // If no move is admissible (shouldn't happen unless all are tabu
            // with no aspiration), pick a random non-tabu or fall through.
            let (var, applied_delta) = match best_var {
                Some(v) => (v, best_delta),
                None => {
                    let v = rng.index(n);
                    let d = delta_cache.deltas[v];
                    (v, d)
                }
            };

            instance.flip_with_term_state(var, &mut sol, &mut term_state);
            obj += applied_delta;
            instance.update_delta_cache_after_flip(var, &sol, &term_state, &mut delta_cache);
            tabu_until[var] = iter + tenure as u64;
            restart_iters += 1;
            iter += 1;

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
                    break;
                }
            }
        }

        total_iters += restart_iters;
        log::debug!(
            "restart {} done: best_in_run = {:.6}, iters = {}",
            restart,
            obj,
            restart_iters
        );

        if timed_out || cutoff_reached || interrupted {
            break;
        }
    }

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

    log::info!(
        "Tabu finished: status={}, obj={}, iters={}, time={:.3}s, tts={:.3}s",
        status,
        best_obj,
        total_iters,
        solving_time,
        best_tts
    );

    let solution = best_sol.expect("Tabu always finds at least one solution");
    let objective = solution.evaluate(instance);

    let result = HeuristicResult {
        method: "Tabu",
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            max_iterations: Some(1_000),
            restarts: 3,
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert!((result.objective).abs() < 1e-10);
        assert_eq!(
            result.solution.to_vec::<f64>(instance.var_type()),
            vec![0.0, 0.0, 0.0]
        );
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
            max_iterations: Some(1_000),
            restarts: 3,
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert!((result.objective - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn cutoff() {
        let mut builder = HuboModel::binary(100);
        for i in 0..100 {
            builder = builder.add_linear(i, 1.0);
        }
        builder = builder.add_quadratic(0, 1, -50.0);
        let instance = builder.build();

        let config = Config {
            common: CommonConfig {
                cutoff: Some(0.0),
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert_eq!(result.status, Status::Cutoff);
        assert!(result.objective <= 0.0);
    }

    #[test]
    fn time_limit() {
        let instance = HuboModel::binary(100)
            .add_terms((0..99).map(|i| (vec![i, i + 1], 1.0)))
            .build();

        let config = Config {
            common: CommonConfig {
                time_limit: Some(0.01),
                seed: Some(42),
                ..Default::default()
            },
            max_iterations: Some(u64::MAX),
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert_eq!(result.status, Status::TimeLimit);
    }

    #[test]
    fn respects_initial_solution() {
        let instance = HuboModel::binary(2)
            .add_linear(0, -1.0)
            .add_linear(1, -1.0)
            .build();
        // instance.initial_solution = Some(vec![1.0, 1.0]);

        let config = Config {
            common: CommonConfig {
                seed: Some(42),
                ..Default::default()
            },
            max_iterations: Some(100),
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert!((result.objective - (-2.0)).abs() < 1e-10);
    }
}
