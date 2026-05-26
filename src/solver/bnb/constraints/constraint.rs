use crate::fixes::Fixes;

pub(crate) enum ConstraintPropagation {
    NoChange,
    Fixed,
    Infeasible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ConstraintCleanup {
    Keep,
    Drop,
    Infeasible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ConstraintKey {
    Parity {
        free_vars: Vec<usize>,
        odd_required: bool,
    },
    Cover {
        items: Vec<Vec<usize>>,
        max_active: usize,
    },
    #[allow(dead_code)]
    LexOrder { vars: Vec<usize> },
    /// Lex-comparison constraint: enforce a ≤_lex p(a) for a generator permutation p.
    /// `pairs` holds (k, p⁻¹[k]) in position order — the first pair where they differ
    /// must satisfy a[k] < a[p⁻¹[k]].
    LexComparison { pairs: Vec<(usize, usize)> },
}

pub(crate) trait ConstraintClone {
    fn clone_box(&self) -> Box<dyn Constraint>;
}

impl<T> ConstraintClone for T
where
    T: 'static + Constraint + Clone,
{
    fn clone_box(&self) -> Box<dyn Constraint> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Constraint> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub(crate) trait Constraint: std::fmt::Debug + ConstraintClone + Send + Sync {
    fn key(&self) -> ConstraintKey;

    fn cleanup_and_check(&self, fixed: &Fixes) -> ConstraintCleanup;

    fn accumulate_branch_scores(&self, fixed: &Fixes, scores: &mut [u64]);

    fn propagate(&self, assigned: &Fixes, fixed_this_round: &mut Fixes) -> ConstraintPropagation;
}
