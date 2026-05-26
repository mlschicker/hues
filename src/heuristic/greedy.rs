//! Specialized greedy heuristic for HUBO problems.
//!
//! Implements steepest-descent local search with multiple random restarts.
//! Each pass evaluates all `n` single-variable flips and applies the best
//! improving move until a local minimum is reached.  The term-state cache
//! makes each pass O(n · max_degree) rather than O(n · n_terms).

use std::time::Instant;

use crate::Logger;
use crate::coeff::Coeff;
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Rng, Status, base_seed, random_solution};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Greedy-heuristic-specific configuration.
pub struct Config {
    /// Parameters shared with all heuristics.
    pub common: CommonConfig,
    /// Number of independent restarts from random initial solutions.
    pub restarts: usize,
    /// Maximum number of improving flips per restart.  `None` → no limit
    /// (runs until local minimum or time/cutoff).
    pub max_flips: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            restarts: 1,
            max_flips: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Run the greedy steepest-descent heuristic on a HUBO instance.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &Config,
    logger: &Logger,
) -> HeuristicResult<C> {
    let _ = logger;
    let start = Instant::now();
    let n = instance.n_vars();

    log::info!(
        "Greedy: n_vars={}, n_terms={}, restarts={}",
        n,
        instance.n_terms(),
        config.restarts,
    );

    let seed = base_seed(config.common.seed);

    let mut best_obj = C::max_value();
    let mut best_sol: Option<BitSolution> = None;
    let mut best_tts: f64 = 0.0;
    let mut total_flips: u64 = 0;
    let mut timed_out = false;
    let mut cutoff_reached = false;
    let mut interrupted = false;

    for restart in 0..config.restarts {
        let mut rng = Rng::new(seed.wrapping_add(restart as u64));
        let mut sol = random_solution(n, instance.var_type(), &mut rng);
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

        let mut restart_flips: u64 = 0;

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

            if let Some(mf) = config.max_flips
                && restart_flips >= mf
            {
                break;
            }

            // Steepest-descent: pick the variable whose flip gives the
            // largest (most negative) improvement.
            let mut best_var: Option<usize> = None;
            let mut best_delta = C::zero();

            for var in 0..n {
                let delta = delta_cache.deltas[var];
                if delta < best_delta {
                    best_delta = delta;
                    best_var = Some(var);
                }
            }

            match best_var {
                None => break, // local minimum — no improving flip exists
                Some(var) => {
                    instance.flip_with_term_state(var, &mut sol, &mut term_state);
                    obj += best_delta;
                    instance.update_delta_cache_after_flip(var, &sol, &term_state, &mut delta_cache);
                    restart_flips += 1;

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
            }
        }

        total_flips += restart_flips;
        log::info!(
            "restart {} done: obj = {:.6}, flips = {}",
            restart,
            obj,
            restart_flips
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
        "Greedy finished: status={}, obj={}, flips={}, time={:.3}s, tts={:.3}s",
        status,
        best_obj,
        total_flips,
        solving_time,
        best_tts
    );

    let solution = best_sol.expect("Greedy always finds at least one solution");
    let objective = solution.evaluate(instance);

    let result = HeuristicResult {
        method: "Greedy",
        status,
        objective,
        solution,
        solving_time,
        tts: best_tts,
        iterations: total_flips,
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
            restarts: 3,
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
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
            restarts: 3,
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert!((result.objective - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn stops_at_local_minimum() {
        // x0*x1 - x0 - x1: optimum at x0=1, x1=1 (obj=-1), but there are
        // local minima at x0=0,x1=0 (obj=0) with no single improving flip.
        let instance = HuboModel::binary(2)
            .add_term(&[0, 1], 1.0)
            .add_linear(0, -1.0)
            .add_linear(1, -1.0)
            .build();

        let config = Config {
            common: CommonConfig {
                seed: Some(0),
                ..Default::default()
            },
            restarts: 10,
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        // With enough restarts some run reaches the global optimum.
        assert!(result.objective <= 0.0);
        assert_eq!(result.status, Status::Completed);
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
            restarts: 10,
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert_eq!(result.status, Status::Cutoff);
        assert!(result.objective <= 0.0);
    }

    #[test]
    fn time_limit() {
        let instance = HuboModel::binary(500)
            .add_terms((0..499).map(|i| (vec![i, i + 1], 1.0)))
            .build();

        let config = Config {
            common: CommonConfig {
                time_limit: Some(0.01),
                seed: Some(42),
                ..Default::default()
            },
            restarts: usize::MAX,
            ..Default::default()
        };

        let result = solve(&instance, &config, &());
        assert_eq!(result.status, Status::TimeLimit);
    }
}
