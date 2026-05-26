use crate::coeff::Coeff;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

use super::Node;

#[derive(Debug, Clone, Copy)]
pub struct Trwbp {
    pub max_iter: usize,
    pub damping: f64,
}

impl Default for Trwbp {
    fn default() -> Self {
        Self {
            max_iter: 8,
            damping: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
struct PairEdge {
    u: usize,
    v: usize,
    table: [[f64; 2]; 2],
}

fn argmin_marginals2(a0: f64, a1: f64) -> [f64; 2] {
    let eps = 1e-12;
    if (a0 - a1).abs() <= eps {
        [0.5, 0.5]
    } else if a0 < a1 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    }
}

fn trwbp_dual_bound(
    constant: f64,
    unary: &[[f64; 2]],
    edges: &[PairEdge],
    cfg: &Trwbp,
) -> f64 {
    let n = unary.len();
    let m = edges.len();
    if n == 0 {
        return constant;
    }

    let mut msg_uv = vec![[0.0_f64; 2]; m];
    let mut msg_vu = vec![[0.0_f64; 2]; m];

    let eval_dual = |msg_uv: &[[f64; 2]], msg_vu: &[[f64; 2]]| -> f64 {
        let mut incoming = vec![[0.0_f64; 2]; n];
        for (eidx, e) in edges.iter().enumerate() {
            incoming[e.u][0] += msg_vu[eidx][0];
            incoming[e.u][1] += msg_vu[eidx][1];
            incoming[e.v][0] += msg_uv[eidx][0];
            incoming[e.v][1] += msg_uv[eidx][1];
        }

        let mut lb = constant;
        for i in 0..n {
            let c0 = unary[i][0] - incoming[i][0];
            let c1 = unary[i][1] - incoming[i][1];
            lb += c0.min(c1);
        }

        for (eidx, e) in edges.iter().enumerate() {
            let mut best = f64::INFINITY;
            for xu in 0..2 {
                for xv in 0..2 {
                    let v = e.table[xu][xv] + msg_uv[eidx][xv] + msg_vu[eidx][xu];
                    if v < best {
                        best = v;
                    }
                }
            }
            lb += best;
        }

        lb
    };

    let mut best_lb = eval_dual(&msg_uv, &msg_vu);
    let mut step_scale = 0.25_f64;

    for it in 0..cfg.max_iter {
        let step = step_scale / (1.0 + it as f64).sqrt();

        let mut incoming = vec![[0.0_f64; 2]; n];
        for (eidx, e) in edges.iter().enumerate() {
            incoming[e.u][0] += msg_vu[eidx][0];
            incoming[e.u][1] += msg_vu[eidx][1];
            incoming[e.v][0] += msg_uv[eidx][0];
            incoming[e.v][1] += msg_uv[eidx][1];
        }

        let mut node_marg = vec![[0.0_f64; 2]; n];
        for i in 0..n {
            let c0 = unary[i][0] - incoming[i][0];
            let c1 = unary[i][1] - incoming[i][1];
            node_marg[i] = argmin_marginals2(c0, c1);
        }

        let mut next_uv = msg_uv.clone();
        let mut next_vu = msg_vu.clone();

        for (eidx, e) in edges.iter().enumerate() {
            let mut edge_val = [[0.0_f64; 2]; 2];
            let mut min_val = f64::INFINITY;
            for xu in 0..2 {
                for xv in 0..2 {
                    let v = e.table[xu][xv] + msg_uv[eidx][xv] + msg_vu[eidx][xu];
                    edge_val[xu][xv] = v;
                    if v < min_val {
                        min_val = v;
                    }
                }
            }

            let eps = 1e-12;
            let mut count = 0.0;
            let mut edge_marg_u = [0.0_f64; 2];
            let mut edge_marg_v = [0.0_f64; 2];
            for xu in 0..2 {
                for xv in 0..2 {
                    if (edge_val[xu][xv] - min_val).abs() <= eps {
                        count += 1.0;
                        edge_marg_u[xu] += 1.0;
                        edge_marg_v[xv] += 1.0;
                    }
                }
            }
            if count > 0.0 {
                edge_marg_u[0] /= count;
                edge_marg_u[1] /= count;
                edge_marg_v[0] /= count;
                edge_marg_v[1] /= count;
            }

            let mut upd_uv = [0.0_f64; 2];
            let mut upd_vu = [0.0_f64; 2];
            for s in 0..2 {
                upd_uv[s] = msg_uv[eidx][s] + step * (edge_marg_v[s] - node_marg[e.v][s]);
                upd_vu[s] = msg_vu[eidx][s] + step * (edge_marg_u[s] - node_marg[e.u][s]);
            }

            let mu = 0.5 * (upd_uv[0] + upd_uv[1]);
            let mv = 0.5 * (upd_vu[0] + upd_vu[1]);
            upd_uv[0] -= mu;
            upd_uv[1] -= mu;
            upd_vu[0] -= mv;
            upd_vu[1] -= mv;

            next_uv[eidx][0] = cfg.damping * msg_uv[eidx][0] + (1.0 - cfg.damping) * upd_uv[0];
            next_uv[eidx][1] = cfg.damping * msg_uv[eidx][1] + (1.0 - cfg.damping) * upd_uv[1];
            next_vu[eidx][0] = cfg.damping * msg_vu[eidx][0] + (1.0 - cfg.damping) * upd_vu[0];
            next_vu[eidx][1] = cfg.damping * msg_vu[eidx][1] + (1.0 - cfg.damping) * upd_vu[1];
        }

        msg_uv = next_uv;
        msg_vu = next_vu;

        let cur_lb = eval_dual(&msg_uv, &msg_vu);
        if cur_lb > best_lb {
            best_lb = cur_lb;
            step_scale = (step_scale * 1.05).min(1.0);
        } else {
            step_scale = (step_scale * 0.7).max(1e-4);
        }
    }

    best_lb
}

fn trwbp_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &Trwbp,
) -> f64 {
    use std::collections::HashMap;

    let mut constant = instance.offset.to_f64();
    let mut unary_x = vec![0.0_f64; instance.n_vars()];
    let mut unary_h = vec![0.0_f64; instance.n_vars()];
    let mut pair_bin: HashMap<(usize, usize), f64> = HashMap::new();
    let mut pair_spin: HashMap<(usize, usize), f64> = HashMap::new();

    for ti in 0..instance.terms.len() {
        let Some(term_status) = &node.term_status[ti] else {
            continue;
        };
        let coeff = term_status.coeff.to_f64();
        match V::VAR_TYPE {
            VarType::Bin => {
                if term_status.free_variables.is_empty() {
                    // Fully assigned
                    constant += coeff;
                } else {
                    // Partially assigned
                    if coeff == 0.0 {
                        continue;
                    }
                    match term_status.free_variables.as_slice() {
                        [] => {
                            constant += coeff;
                        }
                        [i] => {
                            unary_x[*i] += coeff;
                        }
                        [i, j] => {
                            let key = if i < j { (*i, *j) } else { (*j, *i) };
                            *pair_bin.entry(key).or_insert(0.0) += coeff;
                        }
                        _ => {
                            if coeff < 0.0 {
                                constant += coeff;
                            }
                        }
                    }
                }
            }
            VarType::Spin => {
                if term_status.free_variables.is_empty() {
                    // Fully assigned
                    constant += coeff;
                } else {
                    // Partially assigned
                    if coeff == 0.0 {
                        continue;
                    }
                    match term_status.free_variables.as_slice() {
                        [] => {
                            constant += coeff;
                        }
                        [i] => {
                            unary_h[*i] += coeff;
                        }
                        [i, j] => {
                            let key = if i < j { (*i, *j) } else { (*j, *i) };
                            *pair_spin.entry(key).or_insert(0.0) += coeff;
                        }
                        _ => {
                            constant += -coeff.abs();
                        }
                    }
                }
            }
        }
    }

    let mut unary = vec![[0.0_f64; 2]; instance.n_vars()];
    let mut edges: Vec<PairEdge> = Vec::new();

    match V::VAR_TYPE {
        VarType::Bin => {
            for i in 0..instance.n_vars() {
                unary[i] = [0.0, unary_x[i]];
            }
            for ((u, v), c) in pair_bin {
                if c == 0.0 {
                    continue;
                }
                edges.push(PairEdge {
                    u,
                    v,
                    table: [[0.0, 0.0], [0.0, c]],
                });
            }
        }
        VarType::Spin => {
            for i in 0..instance.n_vars() {
                let h = unary_h[i];
                unary[i] = [-h, h];
            }
            for ((u, v), j) in pair_spin {
                if j == 0.0 {
                    continue;
                }
                edges.push(PairEdge {
                    u,
                    v,
                    table: [[j, -j], [-j, j]],
                });
            }
        }
    }

    trwbp_dual_bound(constant, &unary, &edges, cfg)
}

pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
    cfg: &Trwbp,
) -> C {
    let cheap = super::cheap::compute(instance, node);
    let trw = trwbp_lower_bound(instance, node, cfg);
    if !trw.is_finite() {
        return cheap;
    }
    let eps = 1e-9 * (1.0 + trw.abs());
    let trw_c = C::from_f64_lb(trw - eps);
    if trw_c > cheap { trw_c } else { cheap }
}
