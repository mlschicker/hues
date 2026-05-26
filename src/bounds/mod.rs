use std::sync::Arc;

use crate::coeff::Coeff;
use crate::solver::bnb::{Config, Node};
use crate::{domain::VarDomain, instance::HuboInstance};

pub mod cheap;
pub mod cluster_subgradient;
pub mod exact_lasserre;
pub mod hittingset;
pub mod lasserre;
pub mod lp;
pub mod rlt_lagrangian;
mod roof_dual;
pub mod sherali_adams;
pub mod srmp;
pub mod subgradient;
pub mod trwbp;

pub use cheap::Cheap;
pub use cluster_subgradient::ClusterSubgradient;
pub use exact_lasserre::ExactLasserre;
pub use hittingset::HittingSet;
pub use lasserre::{ChordalSdp, Lasserre};
pub use lp::{LpBasis, LpBound};
pub use rlt_lagrangian::RltLp;
pub use sherali_adams::SheraliAdams;
pub use srmp::Srmp;
pub use subgradient::Subgradient;
pub use trwbp::Trwbp;

#[derive(Debug, Clone)]
pub(crate) struct SoftParityEq<C: Coeff> {
    pub(crate) vars: Vec<usize>,
    pub(crate) odd_required: bool,
    pub(crate) penalty: C,
}

pub(crate) use cheap::{lower_bound, lower_bound_spin_base, parity_unsat_core};

pub trait LowerBound: std::fmt::Debug + Clone + Send + Sync + 'static {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent_ub: Option<C>,
    ) -> C;

    /// Like [`compute`] but also returns the first parity-unsat core from the
    /// cheap GE pass for reuse in branching variable selection.
    /// The default delegates to `compute` and returns no core; override in
    /// bounds (like `Cheap`) that can extract the core without extra GE cost.
    fn compute_with_core<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent_ub: Option<C>,
    ) -> (C, Option<Vec<usize>>) {
        (self.compute(instance, node, incumbent_ub), None)
    }

    fn warmstart<C: Coeff, V: VarDomain>(
        &self,
        _instance: &Arc<HuboInstance<C, V>>,
        _node: &mut Node<C>,
    ) {
    }

    /// Return the parity-unsat core hint stored in `node.lb_warm_start`, if any.
    ///
    /// Only meaningful for bounds that record GE cores (currently `Cheap`).
    /// Called by `select_branch_var` to fold the inherited hint into the
    /// branching candidate scoring before the fresh computation's core
    /// (returned from `compute_with_core`) is available or preferred.
    fn branch_hint<C: Coeff>(&self, _node: &Node<C>) -> Option<Vec<usize>> {
        None
    }
}

impl LowerBound for Cheap {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        cheap::compute(instance, node)
    }

    fn compute_with_core<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> (C, Option<Vec<usize>>) {
        cheap::compute_with_core(instance, node)
    }

    fn branch_hint<C: Coeff>(&self, node: &Node<C>) -> Option<Vec<usize>> {
        node.lb_warm_start
            .as_ref()
            .and_then(|ws| ws.downcast_ref::<Vec<usize>>())
            .cloned()
    }
}

impl LowerBound for Trwbp {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        trwbp::compute(instance, node, self).max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for HittingSet {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        let cheap = cheap::compute(instance, node);
        let hs = hittingset::compute(instance, node, self);
        log::debug!("Hitting Set LB: {hs}, Cheap LB: {cheap}");
        hs.max_of(cheap)
    }
}

impl LowerBound for Lasserre {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        lasserre::compute(instance, node, &self.0).max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for ChordalSdp {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        lasserre::compute_chordal(instance, node, &self.0).max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for ExactLasserre {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        exact_lasserre::compute(instance, node, &self.0).max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for Subgradient {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent_ub: Option<C>,
    ) -> C {
        subgradient::compute(instance, node, self, incumbent_ub)
            .max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for Srmp {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent_ub: Option<C>,
    ) -> C {
        let cheap = cheap::compute(instance, node);
        // Only run SRMP at the root node (no variables fixed yet); fall back
        // to cheap everywhere else so inner BnB nodes stay fast.
        if node.fixed.num_free() < instance.n_vars() {
            return cheap;
        }
        srmp::compute(instance, node, self, incumbent_ub).max_of(cheap)
    }
}

impl LowerBound for ClusterSubgradient {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        incumbent_ub: Option<C>,
    ) -> C {
        cluster_subgradient::compute(instance, node, self, incumbent_ub)
            .max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for SheraliAdams {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        let cheap = cheap::compute(instance, node);
        // SA is a full LP relaxation — expensive.  Only run it when no variables
        // are fixed yet (the root node), mirroring the Srmp pattern.  Inner BnB
        // nodes fall back to cheap so the search loop stays fast.
        if node.fixed.num_free() < instance.n_vars() {
            return cheap;
        }
        sherali_adams::compute(instance, node, &self.0).max_of(cheap)
    }
}

impl LowerBound for RltLp {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        rlt_lagrangian::compute(instance, node, &self.0).max_of(cheap::compute(instance, node))
    }
}

impl LowerBound for LpBound {
    fn compute<C: Coeff, V: VarDomain>(
        &self,
        instance: &Arc<HuboInstance<C, V>>,
        node: &mut Node<C>,
        _incumbent_ub: Option<C>,
    ) -> C {
        // Run LP at all nodes; warm-start via parent's simplex basis stored in node.
        lp::compute(instance, node, &self.0).max_of(cheap::compute(instance, node))
    }
}

impl Default for Config<Srmp> {
    fn default() -> Self {
        Self {
            lb: Srmp::default(),
            time_limit: None,
            node_limit: None,
            cutoff: None,
            progress_every_nodes: Some(5_000),
            stats_csv: None,
            instance_name: None,
            solution_file: None,
            warm_start_heuristics: true,
            warm_start_heuristic_time_limit: Some(0.5),
            n_threads: 1,
            node_kernelization: crate::kernelization::KernelizationConfig::default(),
            probing: crate::solver::bnb::ProbingConfig::default(),
            strong_branching: crate::solver::bnb::StrongBranchingConfig::default(),
            optimality_tol: 1e-5,
            kernelization: true,
            seed: 0,
            bound_log_min_improvement_pct: 0.0,
        }
    }
}

impl Default for Config<LpBound> {
    fn default() -> Self {
        Self {
            lb: LpBound::default(),
            time_limit: None,
            node_limit: None,
            cutoff: None,
            progress_every_nodes: Some(5_000),
            stats_csv: None,
            instance_name: None,
            solution_file: None,
            warm_start_heuristics: true,
            warm_start_heuristic_time_limit: Some(0.5),
            n_threads: 1,
            node_kernelization: crate::kernelization::KernelizationConfig::default(),
            probing: crate::solver::bnb::ProbingConfig::default(),
            strong_branching: crate::solver::bnb::StrongBranchingConfig::default(),
            optimality_tol: 1e-5,
            kernelization: true,
            seed: 0,
            bound_log_min_improvement_pct: 0.0,
        }
    }
}

impl Default for Config<Cheap> {
    fn default() -> Self {
        Self {
            lb: Cheap,
            time_limit: None,
            node_limit: None,
            cutoff: None,
            progress_every_nodes: Some(5_000),
            stats_csv: None,
            instance_name: None,
            solution_file: None,
            warm_start_heuristics: true,
            warm_start_heuristic_time_limit: Some(0.5),
            n_threads: 1,
            node_kernelization: crate::kernelization::KernelizationConfig::default(),
            probing: crate::solver::bnb::ProbingConfig::default(),
            strong_branching: crate::solver::bnb::StrongBranchingConfig::default(),
            optimality_tol: 1e-5,
            kernelization: true,
            seed: 0,
            bound_log_min_improvement_pct: 0.0,
        }
    }
}

pub(crate) fn warmstart_lower_bound<C: Coeff, V: VarDomain, Lb: LowerBound>(
    node: &mut Node<C>,
    config: &Config<Lb>,
    instance: &Arc<HuboInstance<C, V>>,
) {
    config.lb.warmstart(instance, node);
}

pub(crate) use roof_dual::apply_roof_dual_fixings;

pub(crate) fn compute_node_lb<C: Coeff, V: VarDomain, Lb: LowerBound>(
    node: &mut Node<C>,
    config: &Config<Lb>,
    incumbent_ub: Option<C>,
    instance: &Arc<HuboInstance<C, V>>,
) -> C {
    let lb = config.lb.compute(instance, node, incumbent_ub);
    instance.round_lower_bound_to_objective_grid(lb)
}

/// Like [`compute_node_lb`] but also returns the first parity-unsat core from
/// the cheap GE pass, so the caller can forward it to `next_branch_var` and
/// skip a redundant Gaussian-elimination run.
pub(crate) fn compute_node_lb_with_core<C: Coeff, V: VarDomain, Lb: LowerBound>(
    node: &mut Node<C>,
    config: &Config<Lb>,
    incumbent_ub: Option<C>,
    instance: &Arc<HuboInstance<C, V>>,
) -> (C, Option<Vec<usize>>) {
    let (lb, core) = config.lb.compute_with_core(instance, node, incumbent_ub);
    (instance.round_lower_bound_to_objective_grid(lb), core)
}
