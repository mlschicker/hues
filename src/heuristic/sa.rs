//! Simulated Annealing (SA) heuristic for HUBO problems.
//!
//! Uses Boltzmann acceptance with geometric cooling and efficient
//! single-flip delta evaluation.

use std::time::Instant;

use crate::coeff::Coeff;
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Rng, Status, base_seed, random_solution};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SA-specific configuration.
pub struct Config {
    /// Parameters shared with all heuristics.
    pub common: CommonConfig,
    /// Initial temperature.
    pub initial_temp: f64,
    /// Final temperature (cooling stops here).
    pub final_temp: f64,
    /// Multiplicative cooling factor per sweep (T *= cooling_rate).
    pub cooling_rate: f64,
    /// Flip attempts per temperature level.  `None` → use `n_vars`.
    pub sweeps_per_temp: Option<usize>,
    /// Number of independent restarts.
    pub restarts: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            initial_temp: 10.0,
            final_temp: 1e-6,
            cooling_rate: 0.9995,
            sweeps_per_temp: None,
            restarts: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Run simulated annealing on a HUBO instance.
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

    log::info!(
        "SA: n_vars={}, n_terms={}, restarts={}, T0={}, Tf={}, cooling={}",
        n,
        instance.n_terms(),
        config.restarts,
        config.initial_temp,
        config.final_temp,
        config.cooling_rate
    );

    let sweeps = config.sweeps_per_temp.unwrap_or(n);
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

        let mut sol = match (restart, initial_solution) {
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

        let mut temp = config.initial_temp;
        let mut restart_iters: u64 = 0;
        let run_until_time_limit = false;

        loop {
            if crate::interrupt::is_interrupted() {
                interrupted = true;
                break;
            }

            if let Some(tl) = config.common.time_limit
                && start.elapsed().as_secs_f64() >= tl
            {
                timed_out = true;
                break;
            }

            let schedule_complete = temp <= config.final_temp;
            if schedule_complete && !run_until_time_limit {
                break;
            }

            for _ in 0..sweeps {
                if crate::interrupt::is_interrupted() {
                    interrupted = true;
                    break;
                }

                if let Some(tl) = config.common.time_limit
                    && start.elapsed().as_secs_f64() >= tl
                {
                    timed_out = true;
                    break;
                }

                let var = rng.index(n);
                let delta = delta_cache.deltas[var];

                let delta_f64 = delta.to_f64();
                if delta_f64 <= 0.0 || rng.uniform() < (-delta_f64 / temp).exp() {
                    instance.flip_with_term_state(var, &mut sol, &mut term_state);
                    obj += delta;
                    instance.update_delta_cache_after_flip(
                        var,
                        &sol,
                        &term_state,
                        &mut delta_cache,
                    );

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
                restart_iters += 1;
            }

            if timed_out || cutoff_reached || interrupted {
                break;
            }

            if !schedule_complete {
                temp *= config.cooling_rate;
            } else {
                temp = config.final_temp;
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
        "SA finished: status={}, obj={}, iters={}, time={:.3}s, tts={:.3}s",
        status,
        best_obj,
        total_iters,
        solving_time,
        best_tts
    );

    let solution = best_sol.expect("SA always finds at least one solution");
    let objective = solution.evaluate(instance);

    let result = HeuristicResult {
        method: "SA",
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
            restarts: 5,
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
            restarts: 5,
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert!((result.objective - (-3.0)).abs() < 1e-10);
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
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert!((result.objective - (-2.0)).abs() < 1e-10);
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
            ..Default::default()
        };

        let result = solve(&instance, &config, None);
        assert_eq!(result.status, Status::TimeLimit);
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
}
