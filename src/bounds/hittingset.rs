use crate::coeff::Coeff;
use crate::util::bitset::BitSet;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

use super::{Node, SoftParityEq, cheap, parity_unsat_core};

#[derive(Debug, Clone, Copy)]
pub struct HittingSet {
    pub max_cores: usize,
    pub max_search_nodes: usize,
}

impl Default for HittingSet {
    fn default() -> Self {
        Self {
            max_cores: 64,
            max_search_nodes: 50_000,
        }
    }
}

fn build_soft_parity_eqs<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
) -> Vec<SoftParityEq<C>> {
    let mut soft_eqs: Vec<SoftParityEq<C>> = Vec::new();

    for i in 0..instance.terms.len() {
        let Some(term_status) = &node.term_status[i] else {
            continue;
        };

        let coeff = term_status.coeff;
        let mut free_vars = term_status.free_variables.clone();

        if free_vars.is_empty() {
            continue;
        }

        if coeff == C::zero() {
            continue;
        }

        free_vars.sort_unstable();
        free_vars.dedup();
        if free_vars.is_empty() {
            continue;
        }

        let odd_required = coeff > C::zero();
        let penalty = (coeff + coeff).abs();
        if penalty <= C::zero() {
            continue;
        }

        soft_eqs.push(SoftParityEq {
            vars: free_vars,
            odd_required,
            penalty,
        });
    }

    soft_eqs
}

fn core_for_active<C: Coeff>(
    soft_eqs: &[SoftParityEq<C>],
    active: &[bool],
    n_vars: usize,
) -> Option<Vec<usize>> {
    let mut active_eqs: Vec<BitSet> = Vec::with_capacity(soft_eqs.len());
    let mut active_map: Vec<usize> = Vec::with_capacity(soft_eqs.len());

    for (i, eq) in soft_eqs.iter().enumerate() {
        if !active[i] || eq.vars.is_empty() {
            continue;
        }
        let mut vars_and_rhs = BitSet::new(n_vars + 1);
        for &v in &eq.vars {
            vars_and_rhs.set(v, true);
        }
        vars_and_rhs.set(n_vars, eq.odd_required);
        active_map.push(i);
        active_eqs.push(vars_and_rhs);
    }

    if active_eqs.len() < 2 {
        return None;
    }

    let core_local = parity_unsat_core(active_eqs, n_vars)?;
    Some(core_local.into_iter().map(|idx| active_map[idx]).collect())
}

fn core_packing_lower_bound(weights: &[f64], cores: &[Vec<usize>]) -> f64 {
    let mut residual = weights.to_vec();
    let mut bonus = 0.0_f64;

    loop {
        let mut progress = false;

        for core in cores {
            if core.is_empty() {
                continue;
            }
            let mut min_w = f64::INFINITY;
            for &v in core {
                min_w = min_w.min(residual[v]);
            }
            if min_w.is_finite() && min_w > 1e-12 {
                bonus += min_w;
                for &v in core {
                    residual[v] -= min_w;
                }
                progress = true;
            }
        }

        if !progress {
            break;
        }
    }

    bonus
}

fn greedy_incumbent(
    weights: &[f64],
    cores: &[Vec<usize>],
    var_to_cores: &[Vec<usize>],
) -> (f64, Vec<bool>) {
    let mut covered = vec![false; cores.len()];
    let mut uncovered = cores.len();
    let mut selected = vec![false; weights.len()];
    let mut cost = 0.0_f64;

    while uncovered > 0 {
        let mut best_var = None;
        let mut best_score = f64::INFINITY;

        for v in 0..weights.len() {
            if selected[v] {
                continue;
            }
            let mut gain = 0usize;
            for &cidx in &var_to_cores[v] {
                if !covered[cidx] {
                    gain += 1;
                }
            }
            if gain == 0 {
                continue;
            }
            let score = weights[v] / gain as f64;
            if score < best_score {
                best_score = score;
                best_var = Some(v);
            }
        }

        let Some(v) = best_var else {
            break;
        };
        selected[v] = true;
        cost += weights[v];
        for &cidx in &var_to_cores[v] {
            if !covered[cidx] {
                covered[cidx] = true;
                uncovered -= 1;
            }
        }
    }

    (cost, selected)
}

fn additional_lb(
    covered: &[bool],
    cores: &[Vec<usize>],
    weights: &[f64],
    selected: &[bool],
) -> f64 {
    let mut lb = 0.0_f64;
    for (cidx, core) in cores.iter().enumerate() {
        if covered[cidx] {
            continue;
        }
        let mut min_w = f64::INFINITY;
        for &v in core {
            if !selected[v] {
                min_w = min_w.min(weights[v]);
            }
        }
        if min_w.is_finite() {
            lb = lb.max(min_w);
        }
    }
    lb
}

fn solve_weighted_hitting_set_exact(
    weights: &[f64],
    cores: &[Vec<usize>],
    max_search_nodes: usize,
) -> Option<(f64, Vec<bool>)> {
    if cores.is_empty() {
        return Some((0.0, vec![false; weights.len()]));
    }

    let mut var_to_cores = vec![Vec::<usize>::new(); weights.len()];
    for (cidx, core) in cores.iter().enumerate() {
        for &v in core {
            var_to_cores[v].push(cidx);
        }
    }

    let (mut incumbent_cost, mut incumbent_sel) = greedy_incumbent(weights, cores, &var_to_cores);
    if !incumbent_cost.is_finite() {
        incumbent_cost = f64::INFINITY;
        incumbent_sel = vec![false; weights.len()];
    }

    struct DfsCtx<'a> {
        weights: &'a [f64],
        cores: &'a [Vec<usize>],
        var_to_cores: &'a [Vec<usize>],
        max_search_nodes: usize,
        search_nodes: usize,
        best_cost: f64,
        best_sel: Vec<bool>,
        aborted: bool,
    }

    impl<'a> DfsCtx<'a> {
        fn dfs(
            &mut self,
            selected: &mut [bool],
            covered: &mut [bool],
            uncovered: usize,
            cur_cost: f64,
        ) {
            if self.aborted {
                return;
            }
            if self.search_nodes >= self.max_search_nodes {
                self.aborted = true;
                return;
            }
            self.search_nodes += 1;

            if uncovered == 0 {
                if cur_cost < self.best_cost {
                    self.best_cost = cur_cost;
                    self.best_sel.copy_from_slice(selected);
                }
                return;
            }

            let lb = additional_lb(covered, self.cores, self.weights, selected);
            if cur_cost + lb >= self.best_cost - 1e-12 {
                return;
            }

            let mut pivot_core_idx = None;
            let mut pivot_size = usize::MAX;
            for (cidx, core) in self.cores.iter().enumerate() {
                if covered[cidx] {
                    continue;
                }
                let sz = core.iter().filter(|&&v| !selected[v]).count();
                if sz < pivot_size {
                    pivot_size = sz;
                    pivot_core_idx = Some(cidx);
                }
            }

            let Some(cidx) = pivot_core_idx else {
                return;
            };

            let mut candidates: Vec<usize> = self.cores[cidx]
                .iter()
                .copied()
                .filter(|&v| !selected[v])
                .collect();
            candidates.sort_by(|&a, &b| self.weights[a].total_cmp(&self.weights[b]));

            for v in candidates {
                let next_cost = cur_cost + self.weights[v];
                if next_cost >= self.best_cost - 1e-12 {
                    continue;
                }

                selected[v] = true;
                let mut changed: Vec<usize> = Vec::new();
                let mut next_uncovered = uncovered;
                for &cc in &self.var_to_cores[v] {
                    if !covered[cc] {
                        covered[cc] = true;
                        changed.push(cc);
                        next_uncovered -= 1;
                    }
                }

                self.dfs(selected, covered, next_uncovered, next_cost);

                for cc in changed {
                    covered[cc] = false;
                }
                selected[v] = false;

                if self.aborted {
                    return;
                }
            }
        }
    }

    let mut selected = vec![false; weights.len()];
    let mut covered = vec![false; cores.len()];
    let uncovered = cores.len();

    let mut ctx = DfsCtx {
        weights,
        cores,
        var_to_cores: &var_to_cores,
        max_search_nodes,
        search_nodes: 0,
        best_cost: incumbent_cost,
        best_sel: incumbent_sel,
        aborted: false,
    };

    ctx.dfs(&mut selected, &mut covered, uncovered, 0.0);

    if ctx.aborted {
        None
    } else {
        Some((ctx.best_cost, ctx.best_sel))
    }
}

fn dedup_and_sort_core(mut core: Vec<usize>) -> Vec<usize> {
    core.sort_unstable();
    core.dedup();
    core
}

fn hitting_set_spin_bonus<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &HittingSet,
) -> f64 {
    let soft_eqs = build_soft_parity_eqs(instance, node);
    if soft_eqs.len() < 2 {
        return 0.0;
    }

    let weights: Vec<f64> = soft_eqs.iter().map(|e| e.penalty.to_f64()).collect();
    let mut cores: Vec<Vec<usize>> = Vec::new();

    // Current best hitting-set solution over discovered cores.
    let mut selected = vec![false; soft_eqs.len()];
    let mut hs_cost = 0.0_f64;

    for _ in 0..cfg.max_cores {
        let active: Vec<bool> = selected.iter().map(|&s| !s).collect();
        let Some(core) = core_for_active(&soft_eqs, &active, instance.n_vars()) else {
            break;
        };
        let core = dedup_and_sort_core(core);
        if core.is_empty() {
            break;
        }
        if !cores.iter().any(|c| c == &core) {
            cores.push(core);
        }

        let Some((cost, sel)) =
            solve_weighted_hitting_set_exact(&weights, &cores, cfg.max_search_nodes)
        else {
            return core_packing_lower_bound(&weights, &cores);
        };

        hs_cost = cost;
        selected = sel;
    }

    hs_cost.max(0.0)
}

pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
    cfg: &HittingSet,
) -> C {
    let cheap_lb = cheap::compute::<C, V>(instance, node);
    if V::VAR_TYPE != VarType::Spin {
        return cheap_lb;
    }

    let base = cheap::lower_bound_spin_base(instance, node).to_f64();
    let bonus = hitting_set_spin_bonus(instance, node, cfg);
    if !bonus.is_finite() {
        return cheap_lb;
    }

    let total = base + bonus;
    if !total.is_finite() {
        return cheap_lb;
    }

    let eps = 1e-9 * (1.0 + total.abs());
    let hs_lb = C::from_f64_lb(total - eps);
    if hs_lb > cheap_lb { hs_lb } else { cheap_lb }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_for_active_allocates_rhs_bit() {
        let soft_eqs = vec![
            SoftParityEq {
                vars: vec![0],
                odd_required: false,
                penalty: 2i64,
            },
            SoftParityEq {
                vars: vec![0],
                odd_required: true,
                penalty: 2i64,
            },
        ];
        let active = vec![true, true];

        let core = core_for_active(&soft_eqs, &active, 1).expect("contradictory core");

        assert_eq!(core, vec![0, 1]);
    }
}
