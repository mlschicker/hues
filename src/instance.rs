//! Core HUBO instance types shared across the crate.

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    BitSolution,
    coeff::Coeff,
    domain::{Bin, Spin, VarDomain, VarType},
    fixes::Fixes,
    state::{DeltaCache, TermState},
    term::Term,
};

// ---------------------------------------------------------------------------
// HuboInstance
// ---------------------------------------------------------------------------

/// A fully parsed HUBO instance over variable domain `V`.
#[derive(Debug, Clone, PartialEq)]
pub struct HuboInstance<C: Coeff, V: VarDomain> {
    pub offset: C,
    pub terms: Vec<Term<C>>,
    pub var_terms: Vec<Vec<usize>>,
    /// GCD of non-constant objective differences, when coefficients are exact integers.
    pub objective_granularity: Option<C>,
    /// One attainable objective value used as the residue for grid rounding.
    pub objective_grid_base: C,
    _marker: PhantomData<fn() -> V>,
}

/// Runtime-tagged wrapper used at parse/CLI boundaries.
#[derive(Debug, Clone)]
pub enum HuboInstanceEnum<C: Coeff> {
    Bin(HuboInstance<C, Bin>),
    Spin(HuboInstance<C, Spin>),
}

impl<C: Coeff> HuboInstanceEnum<C> {
    pub fn var_type(&self) -> VarType {
        match self {
            HuboInstanceEnum::Bin(_) => VarType::Bin,
            HuboInstanceEnum::Spin(_) => VarType::Spin,
        }
    }

    pub fn n_vars(&self) -> usize {
        match self {
            HuboInstanceEnum::Bin(i) => i.n_vars(),
            HuboInstanceEnum::Spin(i) => i.n_vars(),
        }
    }

    pub fn n_terms(&self) -> usize {
        match self {
            HuboInstanceEnum::Bin(i) => i.n_terms(),
            HuboInstanceEnum::Spin(i) => i.n_terms(),
        }
    }

    pub fn offset(&self) -> C {
        match self {
            HuboInstanceEnum::Bin(i) => i.offset,
            HuboInstanceEnum::Spin(i) => i.offset,
        }
    }

    pub fn terms(&self) -> &[Term<C>] {
        match self {
            HuboInstanceEnum::Bin(i) => &i.terms,
            HuboInstanceEnum::Spin(i) => &i.terms,
        }
    }
}

impl<C: Coeff, V: VarDomain> HuboInstance<C, V> {
    /// Build a new instance from raw parts. Recomputes `var_terms`.
    pub fn new(n_vars: usize, offset: C, terms: Vec<Term<C>>) -> Self {
        let var_terms = Self::build_var_terms(n_vars, &terms);
        let objective_granularity = V::objective_granularity(&terms);
        let objective_grid_base = V::objective_grid_base(offset, &terms);
        Self {
            offset,
            terms,
            var_terms,
            objective_granularity,
            objective_grid_base,
            _marker: PhantomData,
        }
    }

    /// Build with pre-computed var_terms.
    pub fn from_parts(
        n_vars: usize,
        offset: C,
        terms: Vec<Term<C>>,
        var_terms: Vec<Vec<usize>>,
    ) -> Self {
        debug_assert_eq!(var_terms.len(), n_vars);
        let objective_granularity = V::objective_granularity(&terms);
        let objective_grid_base = V::objective_grid_base(offset, &terms);
        Self {
            offset,
            terms,
            var_terms,
            objective_granularity,
            objective_grid_base,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn round_lower_bound_to_objective_grid(&self, lb: C) -> C {
        self.objective_granularity
            .map(|g| C::ceil_to_grid(lb, self.objective_grid_base, g))
            .unwrap_or(lb)
    }

    /// Variable domain tag for runtime-typed paths.
    #[inline]
    pub fn var_type(&self) -> VarType {
        V::VAR_TYPE
    }

    /// Number of variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.var_terms.len()
    }

    /// Number of non-constant terms.
    #[inline]
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Build reverse incidence lists from term data.
    pub fn build_var_terms(n_vars: usize, terms: &[Term<C>]) -> Vec<Vec<usize>> {
        let mut var_terms = vec![Vec::new(); n_vars];
        for (ti, term) in terms.iter().enumerate() {
            for &idx in &term.indices {
                var_terms[idx].push(ti);
            }
        }
        var_terms
    }

    /// Apply variable fixes to produce a smaller instance with only free variables.
    ///
    /// Returns the reduced instance and a `new_to_old` index map where
    /// `new_to_old[new_idx] == original_idx`, so solutions on the reduced
    /// instance can be lifted back to the original variable space.
    pub fn apply_fixes(&self, fixes: &Fixes) -> (Self, Vec<usize>) {
        let mut new_to_old: Vec<usize> = Vec::with_capacity(self.n_vars());
        let mut old_to_new: Vec<usize> = vec![usize::MAX; self.n_vars()];

        for (i, old_idx) in old_to_new.iter_mut().enumerate() {
            if fixes.get(i).is_none() {
                *old_idx = new_to_old.len();
                new_to_old.push(i);
            }
        }
        let n_free = new_to_old.len();

        let mut terms: Vec<Term<C>> = Vec::with_capacity(self.n_terms());
        let mut offset = self.offset;

        for term in &self.terms {
            let Some((coeff, free)) =
                V::fold_term_under_fixes::<C>(&term.indices, term.coeff, fixes, &old_to_new)
            else {
                continue;
            };
            if free.is_empty() {
                offset += coeff;
            } else {
                terms.push(Term {
                    indices: free,
                    coeff,
                });
            }
        }

        // Collapse duplicate free-variable index sets by summing their coefficients.
        let mut merged: HashMap<Vec<usize>, C> = HashMap::with_capacity(terms.len());
        for Term { indices, coeff } in terms {
            *merged.entry(indices).or_insert(C::zero()) += coeff;
        }
        let mut terms: Vec<Term<C>> = merged
            .into_iter()
            .filter(|(_, c)| *c != C::zero())
            .map(|(indices, coeff)| Term { indices, coeff })
            .collect();
        terms.sort_unstable_by(|a, b| a.indices.cmp(&b.indices));

        let var_terms = Self::build_var_terms(n_free, &terms);
        let instance = Self::from_parts(n_free, offset, terms, var_terms);
        (instance, new_to_old)
    }

    /// Evaluate the objective value for a bit-encoded solution.
    #[inline]
    pub fn evaluate_bitsol(&self, solution: &BitSolution) -> C {
        V::evaluate_bitsol(self, solution)
    }

    /// Build initial per-term status for a given solution.
    #[inline]
    pub fn init_term_state(&self, solution: &BitSolution) -> TermState<V> {
        V::init_term_state(self, solution)
    }

    /// Compute objective delta for flipping variable `var`.
    #[inline]
    pub fn delta_from_term_state(
        &self,
        var: usize,
        solution: &BitSolution,
        term_state: &TermState<V>,
    ) -> C {
        V::delta_from_term_state(self, var, solution, term_state)
    }

    /// Build full flip-delta cache for the current term state.
    pub fn init_delta_cache(
        &self,
        solution: &BitSolution,
        term_state: &TermState<V>,
    ) -> DeltaCache<C> {
        let mut deltas = Vec::with_capacity(self.n_vars());
        for var in 0..self.n_vars() {
            deltas.push(self.delta_from_term_state(var, solution, term_state));
        }
        DeltaCache {
            deltas,
            marks: vec![0; self.n_vars()],
            mark_epoch: 0,
        }
    }

    /// Refresh only deltas adjacent to `flipped_var`.
    pub fn update_delta_cache_after_flip(
        &self,
        flipped_var: usize,
        solution: &BitSolution,
        term_state: &TermState<V>,
        delta_cache: &mut DeltaCache<C>,
    ) {
        delta_cache.mark_epoch = delta_cache.mark_epoch.wrapping_add(1);

        if delta_cache.mark_epoch == 0 {
            delta_cache.marks.fill(0);
            delta_cache.mark_epoch = 1;
        }

        let epoch = delta_cache.mark_epoch;
        delta_cache.marks[flipped_var] = epoch;

        for &ti in &self.var_terms[flipped_var] {
            for &var in &self.terms[ti].indices {
                delta_cache.marks[var] = epoch;
            }
        }

        for var in 0..self.n_vars() {
            if delta_cache.marks[var] == epoch {
                delta_cache.deltas[var] = self.delta_from_term_state(var, solution, term_state);
            }
        }
    }

    /// Apply a single-variable flip by updating solution and cached term status.
    #[inline]
    pub fn flip_with_term_state(
        &self,
        var: usize,
        solution: &mut BitSolution,
        term_state: &mut TermState<V>,
    ) {
        V::flip_with_term_state(self, var, solution, term_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Bin, Spin};

    #[test]
    fn spin_objective_granularity_uses_superset_formula() {
        let terms = vec![
            Term {
                indices: vec![0],
                coeff: 3i64,
            },
            Term {
                indices: vec![1],
                coeff: 7i64,
            },
            Term {
                indices: vec![0, 1],
                coeff: 5i64,
            },
        ];
        let instance = HuboInstance::<i64, Spin>::new(2, 0, terms);

        assert_eq!(instance.objective_granularity, Some(4));
        assert_eq!(instance.objective_grid_base, -5);
        assert_eq!(instance.round_lower_bound_to_objective_grid(-8), -5);
        assert_eq!(instance.round_lower_bound_to_objective_grid(-9), -9);
    }

    #[test]
    fn binary_objective_granularity_is_gcd_of_monomial_coeffs() {
        let terms = vec![
            Term {
                indices: vec![0],
                coeff: 6i64,
            },
            Term {
                indices: vec![1],
                coeff: 10i64,
            },
            Term {
                indices: vec![0, 1],
                coeff: 4i64,
            },
        ];
        let instance = HuboInstance::<i64, Bin>::new(2, 1, terms);

        assert_eq!(instance.objective_granularity, Some(2));
        assert_eq!(instance.objective_grid_base, 1);
        assert_eq!(instance.round_lower_bound_to_objective_grid(2), 3);
        assert_eq!(instance.round_lower_bound_to_objective_grid(3), 3);
    }

    #[test]
    fn binary_granularity_merges_duplicate_subsets() {
        let terms = vec![
            Term {
                indices: vec![0, 1],
                coeff: 3i64,
            },
            Term {
                indices: vec![1, 0],
                coeff: 5i64,
            },
        ];
        let instance = HuboInstance::<i64, Bin>::new(2, 0, terms);

        // Merged coefficient for {0,1} is 8, so granularity should be 8 — not gcd(3, 5) = 1.
        assert_eq!(instance.objective_granularity, Some(8));
    }

    #[test]
    fn floating_point_binary_instances_have_no_granularity() {
        let terms = vec![Term {
            indices: vec![0, 1],
            coeff: 2.5f64,
        }];
        let instance = HuboInstance::<f64, Bin>::new(2, 0.0, terms);

        assert_eq!(instance.objective_granularity, None);
    }

    #[test]
    fn floating_point_instances_do_not_get_exact_granularity() {
        let terms = vec![Term {
            indices: vec![0],
            coeff: 3.0f64,
        }];
        let instance = HuboInstance::<f64, Spin>::new(1, 0.0, terms);

        assert_eq!(instance.objective_granularity, None);
        assert_eq!(instance.round_lower_bound_to_objective_grid(1.25), 1.25);
    }
}
