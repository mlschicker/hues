use std::collections::HashMap;
use crate::coeff::Coeff;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

use super::Node;

#[derive(Debug, Clone, Copy)]
pub struct Subgradient {
    pub max_iter: usize,
    pub step_size: f64,
    pub step_decay: f64,
    pub optimality_tol: f64,
}

impl Default for Subgradient {
    fn default() -> Self {
        Self {
            max_iter: 64,
            step_size: 1.0,
            step_decay: 1.0,
            optimality_tol: 1e-5,
        }
    }
}

/// Warm-start dual multipliers stored in the node.
/// Each entry is `(global_term_idx, [(var_idx, lambda_val)])`.
pub(crate) type SubgradLambda = Vec<(usize, Vec<(usize, f64)>)>;

/// Finds the minimizer of the local binary term for a given lambda, writing y into the provided slice.
/// Returns the minimum value.
fn min_local_bin(coeff: f64, lambda: &[f64], y: &mut [f64]) -> f64 {
    let mut val_not_all = 0.0_f64;
    let mut any_nonneg = false;
    let mut sum_all = 0.0_f64;
    let mut best_j = 0;
    let mut best_lambda = f64::NEG_INFINITY;

    for (j, (&l, yj)) in lambda.iter().zip(y.iter_mut()).enumerate() {
        sum_all += l;
        if l < 0.0 {
            *yj = 1.0;
            val_not_all += l;
        } else {
            *yj = 0.0;
            any_nonneg = true;
        }
        if l > best_lambda {
            best_lambda = l;
            best_j = j;
        }
    }

    if !any_nonneg {
        y.fill(1.0);
        y[best_j] = 0.0;
        val_not_all = sum_all - best_lambda;
    }

    let val_all = sum_all + coeff;
    if val_all < val_not_all {
        y.fill(1.0);
        val_all
    } else {
        val_not_all
    }
}

/// Finds the minimizer of the local spin term for a given lambda, writing y into the provided slice.
/// Returns the minimum value.
///
/// Uses two linear passes over lambda — first to compute optimal values for both parities
/// without allocating, then to fill y only for the winning parity.
fn min_local_spin(coeff: f64, lambda: &[f64], y: &mut [f64]) -> f64 {
    // Pass 1: compute base_cost, parity of natural minimizer, and flip cost.
    // Natural minimizer sets y[j] = -sign(lambda[j]), giving cost = -sum|lambda[j]|.
    let mut base_cost = 0.0_f64;
    let mut neg_count = 0usize; // number of y[j] = -1 in natural minimizer
    let mut min_abs = f64::INFINITY;
    let mut min_idx = 0usize;

    for (j, &l) in lambda.iter().enumerate() {
        let l_abs = l.abs();
        base_cost -= l_abs;
        // y[j] = -sign(l); y[j] < 0 iff sign(l) > 0 iff l is positive or +0.0
        if !l.is_sign_negative() {
            neg_count += 1;
        }
        if l_abs < min_abs {
            min_abs = l_abs;
            min_idx = j;
        }
    }

    // Natural minimizer parity: even neg_count means product = +1 (target_parity=true).
    let current_even = neg_count.is_multiple_of(2);

    // val_plus: objective when product = +1; val_minus: when product = -1.
    // Flipping the min-|lambda| variable costs +2*min_abs.
    let (val_plus, val_minus) = if current_even {
        (coeff + base_cost, -coeff + base_cost + 2.0 * min_abs)
    } else {
        (coeff + base_cost + 2.0 * min_abs, -coeff + base_cost)
    };

    let want_even = val_plus < val_minus;
    let need_flip = current_even != want_even;

    // Pass 2: fill y with natural minimizer, then flip if needed.
    for (&l, yj) in lambda.iter().zip(y.iter_mut()) {
        *yj = -l.signum();
    }
    if need_flip {
        y[min_idx] = -y[min_idx];
    }

    if want_even { val_plus } else { val_minus }
}

/// Finds the minimizer x^* of the Lagrangian for a given lambda
fn find_x_minimizer<C: Coeff, V: VarDomain>(
    sum_lambda_by_var: &mut [f64],
    lambda: &[Vec<f64>],
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    active_indices: &[usize],
    x_star: &mut [f64],
) -> f64 {
    let n = instance.n_vars();

    sum_lambda_by_var.fill(0.0);

    for (tidx, &ti) in active_indices.iter().enumerate() {
        let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
        for (p, &v) in free_vars.iter().enumerate() {
            sum_lambda_by_var[v] += lambda[tidx][p];
        }
    }

    let mut var_part = 0.0_f64;
    match V::VAR_TYPE {
        VarType::Bin => {
            for i in 0..n {
                let a = sum_lambda_by_var[i];
                if a > 0.0 {
                    x_star[i] = 1.0;
                    var_part -= a;
                } else {
                    x_star[i] = 0.0;
                }
            }
        }
        VarType::Spin => {
            for i in 0..n {
                let a = sum_lambda_by_var[i];
                if a >= 0.0 {
                    x_star[i] = 1.0;
                    var_part -= a;
                } else {
                    x_star[i] = -1.0;
                    var_part += a;
                }
            }
        }
    }

    var_part
}

/// Run Lagrangian subgradient ascent.
fn lagrangian_subgradient_lb<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &Subgradient,
    incumbent_ub: Option<C>,
    warmstart: Option<&[(usize, Vec<(usize, f64)>)]>,
) -> (f64, SubgradLambda) {
    let mut constant = instance.offset.to_f64() + node.offset.to_f64();
    let mut active_indices: Vec<usize> = Vec::new();

    for (ti, term_status) in node
        .term_status
        .iter()
        .enumerate()
        .flat_map(|(i, x)| x.as_ref().map(|v| (i, v)))
    {
        let coeff = term_status.coeff.to_f64();
        if term_status.free_variables.is_empty() {
            constant += coeff;
        } else if coeff != 0.0 {
            active_indices.push(ti);
        }
    }

    if active_indices.is_empty() {
        return (constant, Vec::new());
    }

    let eff_coeff = |ti: usize| -> f64 { node.term_status[ti].as_ref().unwrap().coeff.to_f64() };

    let n = instance.n_vars();

    // Build warmstart lookup: global_term_idx → {var_idx → lambda_val}.
    let warmstart_lookup: HashMap<usize, HashMap<usize, f64>> = warmstart
        .map(|ws| {
            ws.iter()
                .map(|(ti, vars)| (*ti, vars.iter().cloned().collect()))
                .collect()
        })
        .unwrap_or_default();

    // Initialise lambda from warmstart where available, zero otherwise.
    let mut lambda: Vec<Vec<f64>> = active_indices
        .iter()
        .map(|&ti| {
            let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
            if let Some(term_ws) = warmstart_lookup.get(&ti) {
                free_vars
                    .iter()
                    .map(|&v| term_ws.get(&v).copied().unwrap_or(0.0))
                    .collect()
            } else {
                vec![0.0; free_vars.len()]
            }
        })
        .collect();

    // Pre-allocate flat y buffer with per-term offsets to avoid per-iteration allocations.
    let mut y_offsets = Vec::with_capacity(active_indices.len() + 1);
    y_offsets.push(0usize);
    for &ti in &active_indices {
        let arity = node.term_status[ti].as_ref().unwrap().free_variables.len();
        y_offsets.push(y_offsets.last().unwrap() + arity);
    }
    let y_total = *y_offsets.last().unwrap();
    let mut y_flat = vec![0.0_f64; y_total];

    // Pre-allocate working buffers.
    let mut x_star = vec![0.0_f64; n];
    let mut sum_lambda_by_var = vec![0.0_f64; n];

    let mut best_dual = f64::NEG_INFINITY;

    for it in 0..cfg.max_iter.max(1) {
        let var_part = find_x_minimizer(
            &mut sum_lambda_by_var,
            &lambda,
            instance,
            node,
            &active_indices,
            &mut x_star,
        );

        // Compute y minimizers into y_flat (zero allocations).
        let mut term_part = 0.0_f64;
        for (tidx, &ti) in active_indices.iter().enumerate() {
            let y_slice = &mut y_flat[y_offsets[tidx]..y_offsets[tidx + 1]];
            let val = match V::VAR_TYPE {
                VarType::Bin => min_local_bin(eff_coeff(ti), &lambda[tidx], y_slice),
                VarType::Spin => min_local_spin(eff_coeff(ti), &lambda[tidx], y_slice),
            };
            term_part += val;
        }

        let dual_val = constant + var_part + term_part;
        if dual_val > best_dual {
            best_dual = dual_val;
        }

        // Polyak target: primal objective value at the current x assignment.
        let mut primal_val = constant;
        for &ti in &active_indices {
            let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
            let chi = free_vars.iter().fold(1.0_f64, |acc, &v| acc * x_star[v]);
            primal_val += eff_coeff(ti) * chi;
        }

        let mut norm_sq = 0.0_f64;
        for (tidx, &ti) in active_indices.iter().enumerate() {
            let y_slice = &y_flat[y_offsets[tidx]..y_offsets[tidx + 1]];
            let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
            for (p, &v) in free_vars.iter().enumerate() {
                let g = y_slice[p] - x_star[v];
                norm_sq += g * g;
            }
        }

        if norm_sq <= 1e-18 {
            break;
        }

        let beta = cfg.step_size.max(0.0) * cfg.step_decay.clamp(0.0, 1.0).powi(it as i32);
        if beta <= 0.0 {
            break;
        }
        let target_ub = incumbent_ub.map_or(primal_val, |u| primal_val.min(u.to_f64()));
        let gap = (target_ub - dual_val).max(0.0);
        if gap <= cfg.optimality_tol {
            break;
        }
        let alpha = beta * gap / norm_sq;

        for (tidx, &ti) in active_indices.iter().enumerate() {
            let y_slice = &y_flat[y_offsets[tidx]..y_offsets[tidx + 1]];
            let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
            for (p, &v) in free_vars.iter().enumerate() {
                let g = y_slice[p] - x_star[v];
                lambda[tidx][p] += alpha * g;
            }
        }
    }

    // Serialise final lambda by (term_idx, var_idx) for node storage.
    let final_lambda: SubgradLambda = active_indices
        .iter()
        .enumerate()
        .map(|(tidx, &ti)| {
            let free_vars = &node.term_status[ti].as_ref().unwrap().free_variables;
            let var_lambdas: Vec<(usize, f64)> = free_vars
                .iter()
                .enumerate()
                .map(|(p, &v)| (v, lambda[tidx][p]))
                .collect();
            (ti, var_lambdas)
        })
        .collect();

    (best_dual, final_lambda)
}

/// Compute a Lagrangian subgradient bound for the given node.
pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
    cfg: &Subgradient,
    incumbent_ub: Option<C>,
) -> C {
    let warmstart = node
        .lb_warm_start
        .as_ref()
        .and_then(|ws| ws.downcast_ref::<SubgradLambda>())
        .cloned();
    let (dual_lb, final_lambda) =
        lagrangian_subgradient_lb(instance, node, cfg, incumbent_ub, warmstart.as_deref());
    node.lb_warm_start = Some(std::sync::Arc::new(final_lambda));

    let eps = 1e-9 * (1.0 + dual_lb.abs());
    C::from_f64_lb(dual_lb - eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::domain::Bin;
    use crate::solver::bnb::Node;
    use crate::term::Term;

    fn brute_bin_node_opt(instance: &HuboInstance<f64, Bin>, node: &Node<f64>) -> f64 {
        let free: Vec<usize> = (0..instance.n_vars())
            .filter(|&i| !node.fixed.assigned.contains(i))
            .collect();
        let mut best = f64::INFINITY;
        for mask in 0usize..(1usize << free.len()) {
            let mut val = instance.offset + node.offset;
            for term_status in node.term_status.iter().flat_map(|x| x.as_ref()) {
                let chi = term_status.free_variables.iter().all(|&v| {
                    let p = free.iter().position(|&fv| fv == v).unwrap();
                    ((mask >> p) & 1) != 0
                });
                if chi {
                    val += term_status.coeff;
                }
            }
            best = best.min(val);
        }
        best
    }

    #[test]
    fn bin_local_solver_matches_simple_cases() {
        let mut y1 = vec![0.0_f64; 2];
        let v1 = min_local_bin(3.0, &[-2.0, -1.0], &mut y1);
        assert!((v1 + 2.0).abs() < 1e-9);
        assert_eq!(y1, vec![1.0, 0.0]);

        let mut y2 = vec![0.0_f64; 3];
        let v2 = min_local_bin(-5.0, &[0.4, 0.1, 2.0], &mut y2);
        assert!((v2 + 2.5).abs() < 1e-9);
        assert_eq!(y2, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn spin_local_solver_respects_parity_flip() {
        let mut y = vec![0.0_f64; 3];
        let v = min_local_spin(2.0, &[1.0, -3.0, 0.2], &mut y);
        let lin = y[0] * 1.0 + y[1] * -3.0 + y[2] * 0.2;
        let chi = y.iter().product::<f64>();
        let brute = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ]
        .iter()
        .map(|&(a, b, c)| 2.0 * (a * b * c) + a * 1.0 + b * -3.0 + c * 0.2)
        .fold(f64::INFINITY, f64::min);

        assert!((v - (2.0 * chi + lin)).abs() < 1e-9);
        assert!((v - brute).abs() < 1e-9);
    }

    #[test]
    fn bin_subgradient_bound_never_exceeds_bruteforce_on_small_node() {
        let terms = vec![
            Term {
                indices: vec![0, 1, 2],
                coeff: -4.0,
            },
            Term {
                indices: vec![0, 2, 3],
                coeff: 3.0,
            },
            Term {
                indices: vec![1, 3],
                coeff: -2.0,
            },
            Term {
                indices: vec![2, 4],
                coeff: 1.5,
            },
            Term {
                indices: vec![0, 4],
                coeff: -1.25,
            },
        ];
        let mut var_terms = vec![Vec::new(); 5];
        for (ti, term) in terms.iter().enumerate() {
            for &v in &term.indices {
                var_terms[v].push(ti);
            }
        }
        let instance = Arc::new(HuboInstance::<f64, Bin>::from_parts(
            5, 0.0, terms, var_terms,
        ));
        let mut node = Node::root(Arc::clone(&instance), f64::NEG_INFINITY);
        node.set_variable(&instance, 0, true).unwrap();

        let cfg = Subgradient {
            max_iter: 128,
            step_size: 1.0,
            step_decay: 1.0,
            optimality_tol: 1e-9,
        };
        let lb = compute(&instance, &mut node, &cfg, None);
        let opt = brute_bin_node_opt(&instance, &node);
        assert!(lb <= opt + 1e-8, "lb={lb}, opt={opt}");
    }
}
