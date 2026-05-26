//! Cluster-based Lagrangian subgradient lower bound for HUBO.
//!
//! Terms are partitioned into clusters, each with at most `max_cluster_vars`
//! distinct free variables.  For each cluster k we introduce a copy y_k of x
//! restricted to the cluster variables and relax the coupling constraint
//! x_i = y_{k,i} with multipliers λ_{k,i}.
//!
//! The Lagrangian splits into:
//!   - x subproblem: independent per variable, solved in O(n).
//!   - y_k subproblem: a small HUBO over ≤ max_cluster_vars variables,
//!     solved exactly by Gray-code enumeration in O(2^max_cluster_vars).
//!
//! Subgradient ascent (Polyak step) maximises the dual g(λ).

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;

use super::Node;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hard cap on cluster size: 2^MAX must be tractable by Gray-code enumeration.
pub const MAX_CLUSTER_VARS: usize = 20;

#[derive(Debug, Clone, Copy)]
pub struct ClusterSubgradient {
    pub max_iter: usize,
    pub step_size: f64,
    pub step_decay: f64,
    pub optimality_tol: f64,
    /// Maximum number of distinct free variables per cluster (≤ MAX_CLUSTER_VARS).
    pub max_cluster_vars: usize,
}

impl Default for ClusterSubgradient {
    fn default() -> Self {
        Self {
            max_iter: 64,
            step_size: 1.0,
            step_decay: 1.0,
            optimality_tol: 1e-5,
            max_cluster_vars: 15,
        }
    }
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

struct Cluster {
    /// Indices into `node.term_status` that belong to this cluster.
    term_indices: Vec<usize>,
    /// Sorted union of free-variable indices across all terms in the cluster.
    vars: Vec<usize>,
}

/// Greedily pack active terms into clusters (First-Fit Decreasing by term size).
/// Terms with more free variables than `max_vars` are returned in `skipped`.
fn build_clusters<C: Coeff>(node: &Node<C>, max_vars: usize) -> (Vec<Cluster>, Vec<usize>) {
    // Collect active term indices (Some entries with non-empty free vars and non-zero coeff).
    let mut active: Vec<usize> = node
        .term_status
        .iter()
        .enumerate()
        .filter_map(|(ti, ts)| {
            ts.as_ref()
                .filter(|t| !t.free_variables.is_empty() && t.coeff.to_f64() != 0.0)
                .map(|_| ti)
        })
        .collect();

    // FFD: largest terms first so big terms claim their own cluster before small ones fill gaps.
    active.sort_unstable_by(|&a, &b| {
        let na = node.term_status[a].as_ref().unwrap().free_variables.len();
        let nb = node.term_status[b].as_ref().unwrap().free_variables.len();
        nb.cmp(&na)
    });

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut skipped: Vec<usize> = Vec::new();

    'outer: for ti in active {
        let term_vars = &node.term_status[ti].as_ref().unwrap().free_variables;

        if term_vars.len() > max_vars {
            skipped.push(ti);
            continue;
        }

        // Try to extend an existing cluster without exceeding max_vars.
        for cluster in &mut clusters {
            let mut new_vars = 0usize;
            for &v in term_vars {
                if cluster.vars.binary_search(&v).is_err() {
                    new_vars += 1;
                }
            }
            if cluster.vars.len() + new_vars <= max_vars {
                cluster.term_indices.push(ti);
                for &v in term_vars {
                    if let Err(pos) = cluster.vars.binary_search(&v) {
                        cluster.vars.insert(pos, v);
                    }
                }
                continue 'outer;
            }
        }

        // Open a new cluster.
        clusters.push(Cluster {
            term_indices: vec![ti],
            vars: term_vars.to_vec(), // already sorted
        });
    }

    (clusters, skipped)
}

// ---------------------------------------------------------------------------
// Gray-code minimization for the y_k subproblem
// ---------------------------------------------------------------------------

/// BIN: minimize over y ∈ {0,1}^n the objective defined by `terms` + `offset`.
/// Returns (min_value, bit-pattern of best assignment).
/// Bit j = 1 means y_j = 1.
fn gray_code_min_bin(
    n: usize,
    terms: &[(f64, Vec<u8>)],
    var_terms: &[Vec<u32>],
    offset: f64,
) -> (f64, u32) {
    debug_assert!(n <= 32 && n > 0);
    let n_terms = terms.len();
    let total = 1u32 << n;
    let mut counts = vec![0u8; n_terms];
    let mut cur = offset;
    let mut best = cur;
    let mut best_state = 0u32;
    let mut state = 0u32;

    for k in 1..total {
        let v = k.trailing_zeros() as usize;
        if (state >> v) & 1 == 0 {
            // 0 → 1
            for &ti in &var_terms[v] {
                let ti = ti as usize;
                counts[ti] += 1;
                if counts[ti] as usize == terms[ti].1.len() {
                    cur += terms[ti].0;
                }
            }
        } else {
            // 1 → 0
            for &ti in &var_terms[v] {
                let ti = ti as usize;
                if counts[ti] as usize == terms[ti].1.len() {
                    cur -= terms[ti].0;
                }
                counts[ti] -= 1;
            }
        }
        state ^= 1u32 << v;
        if cur < best {
            best = cur;
            best_state = state;
        }
    }

    (best, best_state)
}

/// SPIN: minimize over y ∈ {-1,+1}^n.
/// Initial state: all y_j = -1 (bit j = 0).
/// Bit j = 1 means y_j = +1.
fn gray_code_min_spin(
    n: usize,
    terms: &[(f64, Vec<u8>)],
    var_terms: &[Vec<u32>],
    offset: f64,
) -> (f64, u32) {
    debug_assert!(n <= 32 && n > 0);
    let total = 1u32 << n;
    // Initial contribution of term t: coeff * (-1)^|vars| (all vars = -1).
    let mut contrib: Vec<f64> = terms
        .iter()
        .map(|(c, vars)| if vars.len() % 2 == 0 { *c } else { -*c })
        .collect();
    let mut cur = offset + contrib.iter().copied().sum::<f64>();
    let mut best = cur;
    let mut best_state = 0u32;
    let mut state = 0u32;

    for k in 1..total {
        let v = k.trailing_zeros() as usize;
        // Flipping y_v negates every term that contains v.
        for &ti in &var_terms[v] {
            let ti = ti as usize;
            let old = contrib[ti];
            let new = -old;
            contrib[ti] = new;
            cur += new - old;
        }
        state ^= 1u32 << v;
        if cur < best {
            best = cur;
            best_state = state;
        }
    }

    (best, best_state)
}

/// Solve the y_k HUBO subproblem by Gray-code enumeration.
///
/// Returns `(min_val, y_star)` where `y_star[j]` is the optimal value for
/// `cluster.vars[j]` (0.0/1.0 for BIN, -1.0/+1.0 for SPIN).
fn solve_cluster_y(
    cluster: &Cluster,
    term_coeffs: &[f64],
    term_local_vars: &[Vec<u8>], // local var indices for each term in the cluster
    lambda: &[f64],
    var_type: VarType,
) -> (f64, Vec<f64>) {
    let m = cluster.vars.len();

    // Build term list: original HUBO terms + arity-1 linear λ terms.
    // The linear term for variable j is: -λ_j · y_j  (coeff = -λ_j, vars = [j]).
    let mut terms: Vec<(f64, Vec<u8>)> = Vec::with_capacity(term_coeffs.len() + m);

    for (i, &c) in term_coeffs.iter().enumerate() {
        if c != 0.0 {
            terms.push((c, term_local_vars[i].clone()));
        }
    }
    for (j, &lam) in lambda.iter().enumerate() {
        if lam != 0.0 {
            terms.push((-lam, vec![j as u8]));
        }
    }

    if terms.is_empty() {
        return (0.0, vec![if var_type == VarType::Bin { 0.0 } else { -1.0 }; m]);
    }

    // Build adjacency var_terms[v] = list of term indices containing local var v.
    let mut var_terms: Vec<Vec<u32>> = vec![Vec::new(); m];
    for (ti, (_, vars)) in terms.iter().enumerate() {
        for &v in vars {
            var_terms[v as usize].push(ti as u32);
        }
    }

    let (opt_val, best_state) = match var_type {
        VarType::Bin => gray_code_min_bin(m, &terms, &var_terms, 0.0),
        VarType::Spin => gray_code_min_spin(m, &terms, &var_terms, 0.0),
    };

    let y: Vec<f64> = (0..m)
        .map(|j| match var_type {
            VarType::Bin => if (best_state >> j) & 1 == 1 { 1.0 } else { 0.0 },
            VarType::Spin => if (best_state >> j) & 1 == 1 { 1.0 } else { -1.0 },
        })
        .collect();

    (opt_val, y)
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

fn cluster_subgradient_lb<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &ClusterSubgradient,
    incumbent_ub: Option<C>,
) -> f64 {
    // Constant: instance offset + node offset + fully-resolved terms.
    let mut constant = instance.offset.to_f64() + node.offset.to_f64();
    for ts in node.term_status.iter().flatten() {
        if ts.free_variables.is_empty() {
            constant += ts.coeff.to_f64();
        }
    }

    let (clusters, skipped) = build_clusters(node, cfg.max_cluster_vars.min(MAX_CLUSTER_VARS));

    if clusters.is_empty() && skipped.is_empty() {
        return constant;
    }

    let var_type = V::VAR_TYPE;
    let n = instance.n_vars();

    // Cheap bound for unclustered terms (remains fixed across iterations).
    let skipped_cheap: f64 = skipped
        .iter()
        .map(|&ti| {
            let c = node.term_status[ti].as_ref().unwrap().coeff.to_f64();
            match var_type {
                VarType::Bin => c.min(0.0),
                VarType::Spin => -c.abs(),
            }
        })
        .sum();

    // Pre-cache per-cluster term data (coefficients + local var index vecs).
    // Local index of global variable v in cluster k: position in cluster.vars.
    let cluster_data: Vec<(Vec<f64>, Vec<Vec<u8>>)> = clusters
        .iter()
        .map(|cluster| {
            let coeffs: Vec<f64> = cluster
                .term_indices
                .iter()
                .map(|&ti| node.term_status[ti].as_ref().unwrap().coeff.to_f64())
                .collect();

            let local_vars: Vec<Vec<u8>> = cluster.term_indices.iter().map(|&ti| {
                node.term_status[ti]
                    .as_ref()
                    .unwrap()
                    .free_variables
                    .iter()
                    .map(|&v| {
                        // Binary search is O(log m), m ≤ max_cluster_vars.
                        cluster.vars.binary_search(&v).unwrap() as u8
                    })
                    .collect()
            }).collect();

            (coeffs, local_vars)
        })
        .collect();

    // λ_{k,j}: multiplier for cluster k, variable cluster.vars[j].
    let mut lambdas: Vec<Vec<f64>> = clusters
        .iter()
        .map(|c| vec![0.0_f64; c.vars.len()])
        .collect();

    let mut sum_lambda = vec![0.0_f64; n];
    let mut best_dual = f64::NEG_INFINITY;

    for it in 0..cfg.max_iter.max(1) {
        // ── x subproblem ────────────────────────────────────────────────────
        // Each free variable i gets total coefficient Σ_k λ_{k,i}.
        sum_lambda.fill(0.0);
        for (k, cluster) in clusters.iter().enumerate() {
            for (j, &v) in cluster.vars.iter().enumerate() {
                sum_lambda[v] += lambdas[k][j];
            }
        }

        let mut x_star = vec![0.0_f64; n];
        let mut x_part = 0.0_f64;
        match var_type {
            VarType::Bin => {
                for i in 0..n {
                    if sum_lambda[i] < 0.0 {
                        x_star[i] = 1.0;
                        x_part += sum_lambda[i];
                    }
                    // else x_star[i] = 0.0, contribution = 0
                }
            }
            VarType::Spin => {
                for i in 0..n {
                    // minimise sum_lambda[i] * x_i: pick x_i = -sign(sum_lambda[i])
                    if sum_lambda[i] > 0.0 {
                        x_star[i] = -1.0;
                    } else {
                        x_star[i] = 1.0;
                    }
                    x_part -= sum_lambda[i].abs();
                }
            }
        }

        // ── y_k subproblems ─────────────────────────────────────────────────
        let mut cluster_vals = Vec::with_capacity(clusters.len());
        let mut cluster_y: Vec<Vec<f64>> = Vec::with_capacity(clusters.len());

        for (k, cluster) in clusters.iter().enumerate() {
            let (coeffs, local_vars) = &cluster_data[k];
            let (val, y) = solve_cluster_y(cluster, coeffs, local_vars, &lambdas[k], var_type);
            cluster_vals.push(val);
            cluster_y.push(y);
        }

        let cluster_sum: f64 = cluster_vals.iter().sum();
        let dual_val = constant + skipped_cheap + x_part + cluster_sum;

        if dual_val > best_dual {
            best_dual = dual_val;
        }

        // ── Polyak step ─────────────────────────────────────────────────────
        // Primal: evaluate f(x*) over all active terms.
        let mut primal_val = constant;
        for ts in node.term_status.iter().flatten() {
            if ts.free_variables.is_empty() {
                continue;
            }
            let c = ts.coeff.to_f64();
            // product of x* over the free variables in this term
            let prod = ts
                .free_variables
                .iter()
                .fold(1.0_f64, |acc, &v| acc * x_star[v]);
            primal_val += c * prod;
        }

        // Subgradient: g_{k,j} = x*_{vars[j]} - y*_{k,j}
        let mut norm_sq = 0.0_f64;
        for (k, cluster) in clusters.iter().enumerate() {
            for (j, &v) in cluster.vars.iter().enumerate() {
                let g = x_star[v] - cluster_y[k][j];
                norm_sq += g * g;
            }
        }

        if norm_sq <= 1e-18 {
            break; // feasible coupling → dual = primal, stop
        }

        let beta = cfg.step_size * cfg.step_decay.clamp(0.0, 1.0).powi(it as i32);
        if beta <= 0.0 {
            break;
        }

        let target = incumbent_ub.map_or(primal_val, |u| primal_val.min(u.to_f64()));
        let gap = (target - dual_val).max(0.0);
        if gap <= cfg.optimality_tol {
            break;
        }

        let alpha = beta * gap / norm_sq;
        for (k, cluster) in clusters.iter().enumerate() {
            for (j, &v) in cluster.vars.iter().enumerate() {
                lambdas[k][j] += alpha * (x_star[v] - cluster_y[k][j]);
            }
        }
    }

    best_dual
}

/// Compute the cluster-subgradient lower bound for a BnB node.
pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &ClusterSubgradient,
    incumbent_ub: Option<C>,
) -> C {
    let lb = cluster_subgradient_lb(instance, node, cfg, incumbent_ub);
    let eps = 1e-9 * (1.0 + lb.abs());
    C::from_f64_lb(lb - eps)
}

// ---------------------------------------------------------------------------
// LowerBound impl lives in bounds/mod.rs — see there.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::domain::Bin;
    use crate::instance::HuboInstance;
    use crate::solver::bnb::Node;
    use crate::term::Term;
    use super::*;

    fn make_bin(n: usize, terms: Vec<(Vec<usize>, f64)>) -> Arc<HuboInstance<f64, Bin>> {
        let terms = terms
            .into_iter()
            .map(|(idx, c)| Term { indices: idx, coeff: c })
            .collect();
        Arc::new(HuboInstance::new(n, 0.0, terms))
    }

    fn brute_min_bin(instance: &HuboInstance<f64, Bin>) -> f64 {
        let n = instance.n_vars();
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << n) {
            let mut val = instance.offset;
            for term in &instance.terms {
                if term.indices.iter().all(|&i| (mask >> i) & 1 == 1) {
                    val += term.coeff;
                }
            }
            if val < best {
                best = val;
            }
        }
        best
    }

    #[test]
    fn cluster_lb_never_exceeds_optimum() {
        let inst = make_bin(
            6,
            vec![
                (vec![0, 1, 2], -4.0),
                (vec![1, 2, 3], 3.0),
                (vec![3, 4, 5], -2.0),
                (vec![0, 3], 1.5),
                (vec![2, 4], -1.0),
                (vec![1, 5], 2.0),
            ],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = ClusterSubgradient {
            max_iter: 128,
            max_cluster_vars: 4,
            ..Default::default()
        };
        let lb = compute(inst.as_ref(), &node, &cfg, None);
        let opt = brute_min_bin(inst.as_ref());
        assert!(
            lb.to_f64() <= opt + 1e-6,
            "lb={} must not exceed opt={}",
            lb,
            opt
        );
    }

    #[test]
    fn cluster_lb_positive_on_purely_positive_instance() {
        // All positive coefficients → min is 0 (all vars = 0 in BIN).
        let inst = make_bin(
            4,
            vec![(vec![0, 1], 3.0), (vec![2, 3], 2.0), (vec![0, 2, 3], 1.0)],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = ClusterSubgradient { max_iter: 32, ..Default::default() };
        let lb = compute(inst.as_ref(), &node, &cfg, None);
        assert!(lb.to_f64() <= 0.0 + 1e-6, "lb={} should be ≤ 0", lb);
    }
}
