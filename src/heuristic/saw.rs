//! Self-Avoiding Walk (SAW) heuristic for HUBO problems.
//!
//! Performs a random walk on the binary hypercube (each step flips one
//! variable) with the constraint that no variable may be re-flipped until
//! all `n` variables have been visited.  Once a complete "segment" of `n`
//! steps is taken the forbidden set resets and a new segment begins.
//!
//! Compared with tabu search, SAW picks moves *uniformly at random* from
//! the admissible set rather than greedily, making it a diversification-
//! first strategy.  An optional steepest-descent polishing step can be run
//! at the end of each segment to turn each explored region into a local
//! optimum before continuing.

use std::time::Instant;

use crate::Logger;
use crate::coeff::Coeff;
use crate::{domain::VarDomain, instance::HuboInstance};

use super::{BitSolution, CommonConfig, HeuristicResult, Rng, Status, base_seed, random_solution};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SAW-specific configuration.
pub struct Config {
    /// Parameters shared with all heuristics.
    pub common: CommonConfig,
    /// Number of independent walks (restarts from fresh random solutions).
    pub n_walks: usize,
    /// Maximum steps (single-variable flips) per walk.
    /// `None` → unbounded (stops only on time limit, cutoff, or interrupt).
    pub max_steps: Option<u64>,
    /// If `true`, apply steepest-descent local search at the end of each
    /// complete segment (after all `n` variables have been visited once)
    /// before resetting the forbidden set.
    pub local_search: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            n_walks: 1,
            max_steps: None,
            local_search: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Run the self-avoiding walk heuristic on a HUBO instance.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &Config,
    logger: &Logger,
) -> HeuristicResult<C> {
    let _ = logger;
    let start = Instant::now();
    let n = instance.n_vars();

    log::info!(
        "SAW: n_vars={}, n_terms={}, n_walks={}, local_search={}",
        n,
        instance.n_terms(),
        config.n_walks,
        config.local_search,
    );

    let seed = base_seed(config.common.seed);

    let mut best_obj = C::max_value();
    let mut best_sol: Option<BitSolution> = None;
    let mut best_tts: f64 = 0.0;
    let mut total_steps: u64 = 0;
    let mut timed_out = false;
    let mut cutoff_reached = false;
    let mut interrupted = false;

    for walk in 0..config.n_walks {
        let mut rng = Rng::new(seed.wrapping_add(walk as u64));
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

        log::debug!("walk {}: initial obj = {}", walk, obj);

        // Admissible variables for the current segment: maintained as a
        // shuffled deck so we can pick a random element and remove it in
        // O(1) via swap-and-pop.
        let mut admissible: Vec<usize> = (0..n).collect();
        let mut walk_steps: u64 = 0;

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

            if let Some(ms) = config.max_steps
                && walk_steps >= ms
            {
                break;
            }

            // All variables visited in this segment → start a new one.
            if admissible.is_empty() {
                if config.local_search {
                    // Polish the current solution with steepest descent.
                    loop {
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
                            None => break,
                            Some(var) => {
                                instance.flip_with_term_state(var, &mut sol, &mut term_state);
                                obj += best_delta;
                                instance.update_delta_cache_after_flip(
                                    var,
                                    &sol,
                                    &term_state,
                                    &mut delta_cache,
                                );
                            }
                        }
                    }

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

                // Reset the admissible deck for the next segment.
                admissible = (0..n).collect();
            }

            // Pick a uniformly random admissible variable via swap-and-pop.
            let idx = rng.index(admissible.len());
            let var = admissible[idx];
            let last = admissible.len() - 1;
            admissible.swap(idx, last);
            admissible.pop();

            let delta = delta_cache.deltas[var];
            instance.flip_with_term_state(var, &mut sol, &mut term_state);
            obj += delta;
            instance.update_delta_cache_after_flip(var, &sol, &term_state, &mut delta_cache);
            walk_steps += 1;

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

        total_steps += walk_steps;
        log::info!(
            "walk {} done: obj = {:.6}, steps = {}",
            walk,
            obj,
            walk_steps
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
        "SAW finished: status={}, obj={}, steps={}, time={:.3}s, tts={:.3}s",
        status,
        best_obj,
        total_steps,
        solving_time,
        best_tts
    );

    let solution = best_sol.expect("SAW always finds at least one solution");
    let objective = solution.evaluate(instance);

    let result = HeuristicResult {
        method: "SAW",
        status,
        objective,
        solution,
        solving_time,
        tts: best_tts,
        iterations: total_steps,
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
            n_walks: 5,
            max_steps: Some(500),
            local_search: true,
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
            n_walks: 5,
            max_steps: Some(500),
            local_search: true,
        };

        let result = solve(&instance, &config, &());
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
            n_walks: 10,
            max_steps: None,
            local_search: true,
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
            n_walks: usize::MAX,
            max_steps: None,
            local_search: false,
        };

        let result = solve(&instance, &config, &());
        assert_eq!(result.status, Status::TimeLimit);
    }

    #[test]
    fn segment_reset_continues_walk() {
        // Verify that after n steps the forbidden set resets and the walk
        // continues beyond n steps.
        let instance = HuboModel::binary(4)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_linear(2, 1.0)
            .add_linear(3, 1.0)
            .build();

        let config = Config {
            common: CommonConfig {
                seed: Some(7),
                ..Default::default()
            },
            n_walks: 1,
            max_steps: Some(20), // more than n=4
            local_search: false,
        };

        let result = solve(&instance, &config, &());
        assert_eq!(result.iterations, 20);
        assert_eq!(result.status, Status::Completed);
    }
}
