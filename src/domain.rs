//! Variable domains and domain-specific objective operations.

use std::collections::{BTreeMap, HashMap};

use crate::{BitSolution, coeff::Coeff, fixes::Fixes, instance::HuboInstance, state::TermState};

/// Runtime tag used at I/O boundaries (file headers, formatting, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// x_i \in {0, 1}
    Bin,
    /// s_i \in {-1, +1}
    Spin,
}

/// Compile-time variable domain. Implemented by [`Bin`] and [`Spin`].
pub trait VarDomain:
    Copy + Clone + std::fmt::Debug + PartialEq + Eq + Send + Sync + 'static
{
    const VAR_TYPE: VarType;
    /// Per-term auxiliary state stored in [`TermState`].
    type TermStateData: Send + Sync + Clone + std::fmt::Debug + PartialEq;

    /// Reduce a list of variable indices according to the domain.
    fn reduce_indices(indices: &[usize]) -> Vec<usize>;

    /// Apply variable fixes to a single term, possibly producing a smaller term.
    /// Returns `(coeff, free_indices)` after applying fixes; `free_indices` may
    /// be empty (meaning the term collapsed to a constant) and the returned
    /// coefficient is the contribution to the offset in that case.
    /// Returns `None` if the term is killed (BIN with a fixed-to-0 variable).
    fn fold_term_under_fixes<C: Coeff>(
        term_indices: &[usize],
        coeff: C,
        fixes: &Fixes,
        old_to_new: &[usize],
    ) -> Option<(C, Vec<usize>)>;

    /// Evaluate the objective for a complete bit-encoded solution.
    fn evaluate_bitsol<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> C;

    /// Build initial per-term state for a given solution.
    fn init_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        sol: &BitSolution,
    ) -> TermState<Self>;

    /// Compute the objective delta for flipping `var`.
    fn delta_from_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        sol: &BitSolution,
        term_state: &TermState<Self>,
    ) -> C;

    /// Apply a single-variable flip by updating the solution and term state.
    fn flip_with_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        sol: &mut BitSolution,
        term_state: &mut TermState<Self>,
    );

    /// Compute every local field h[j] = -∂E/∂σ_j for a given solution.
    fn local_fields<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> Vec<C>;

    /// Format a single domain value as a single character ('0'/'1' or '+'/'-').
    fn format_char(high: bool) -> char;

    /// Convert a "high" bit to its domain value as a coefficient.
    fn high_to_coeff<C: Coeff>(high: bool) -> C;

    /// Default value (low) used for missing entries when reading a solution file.
    fn default_low<C: Coeff>() -> C {
        Self::high_to_coeff::<C>(false)
    }

    /// Validate a coefficient value against the domain.
    fn is_valid_value<C: Coeff>(value: C) -> bool;

    /// Update a partially assigned term when fixing `variable` to `high`.
    /// Returns `Some(coeff)` if the term resolves to a constant, `None` otherwise.
    fn update_partial_term<C: Coeff>(
        coeff: &mut C,
        free_variables: &mut Vec<usize>,
        variable: usize,
        high: bool,
    ) -> Option<C>;

    /// Variable label letter used in solution files: 'x' for BIN, 's' for SPIN.
    fn var_letter() -> char;

    /// Exact objective-value granularity for this domain, when one is known.
    ///
    /// Domains without an exact integer objective grid return `None`.
    fn objective_granularity<C: Coeff>(_terms: &[crate::term::Term<C>]) -> Option<C> {
        None
    }

    /// One attainable objective value used as the residue for grid rounding.
    fn objective_grid_base<C: Coeff>(offset: C, _terms: &[crate::term::Term<C>]) -> C {
        offset
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bin;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Spin;

impl VarDomain for Bin {
    const VAR_TYPE: VarType = VarType::Bin;
    type TermStateData = Vec<usize>; // active_counts per term

    fn reduce_indices(indices: &[usize]) -> Vec<usize> {
        let mut s: Vec<usize> = indices.to_vec();
        s.sort_unstable();
        s.dedup();
        s
    }

    fn fold_term_under_fixes<C: Coeff>(
        term_indices: &[usize],
        coeff: C,
        fixes: &Fixes,
        old_to_new: &[usize],
    ) -> Option<(C, Vec<usize>)> {
        // Any fixed-to-0 variable kills the whole monomial.
        if term_indices.iter().any(|&i| fixes.get(i) == Some(false)) {
            return None;
        }
        let free: Vec<usize> = term_indices
            .iter()
            .filter(|&&i| fixes.get(i).is_none())
            .map(|&i| old_to_new[i])
            .collect();
        Some((coeff, free))
    }

    fn evaluate_bitsol<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> C {
        let mut value = instance.offset;
        for term in &instance.terms {
            if term.indices.iter().all(|&j| sol.values.contains(j)) {
                value += term.coeff;
            }
        }
        value
    }

    fn init_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        sol: &BitSolution,
    ) -> TermState<Self> {
        let mut term_status = Vec::with_capacity(instance.n_terms());
        let mut active_counts = Vec::with_capacity(instance.n_terms());
        for term in &instance.terms {
            let count = term
                .indices
                .iter()
                .filter(|&&j| sol.values.contains(j))
                .count();
            active_counts.push(count);
            let is_active = count == term.indices.len();
            term_status.push(is_active);
        }
        TermState {
            term_status,
            data: active_counts,
        }
    }

    fn delta_from_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        sol: &BitSolution,
        term_state: &TermState<Self>,
    ) -> C {
        let mut delta = C::zero();
        let var_is_one = sol.values.contains(var);
        let active_counts = &term_state.data;
        for &ti in &instance.var_terms[var] {
            let term = &instance.terms[ti];
            let count = active_counts[ti];
            let len = term.indices.len();
            if var_is_one {
                if count == len {
                    delta -= term.coeff;
                }
            } else if count + 1 == len {
                delta += term.coeff;
            }
        }
        delta
    }

    fn flip_with_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        sol: &mut BitSolution,
        term_state: &mut TermState<Self>,
    ) {
        let was_one = sol.values.contains(var);
        sol.flip(var);
        let active_counts = &mut term_state.data;
        for &ti in &instance.var_terms[var] {
            if was_one {
                active_counts[ti] -= 1;
            } else {
                active_counts[ti] += 1;
            }
            let len = instance.terms[ti].indices.len();
            term_state.term_status[ti] = active_counts[ti] == len;
        }
    }

    fn local_fields<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> Vec<C> {
        let mut h = vec![C::zero(); instance.n_vars()];
        for term in &instance.terms {
            let set_count = term
                .indices
                .iter()
                .filter(|&&i| sol.values.contains(i))
                .count();
            for &j in &term.indices {
                let j_is_set = sol.values.contains(j) as usize;
                if set_count - j_is_set == term.indices.len() - 1 {
                    h[j] += term.coeff;
                }
            }
        }
        h
    }

    fn format_char(high: bool) -> char {
        if high { '1' } else { '0' }
    }

    fn high_to_coeff<C: Coeff>(high: bool) -> C {
        if high { C::one() } else { C::zero() }
    }

    fn is_valid_value<C: Coeff>(value: C) -> bool {
        value == C::zero() || value == C::one()
    }

    fn update_partial_term<C: Coeff>(
        coeff: &mut C,
        free_variables: &mut Vec<usize>,
        variable: usize,
        high: bool,
    ) -> Option<C> {
        if !high {
            *coeff = C::zero();
        }
        free_variables.retain(|&v| v != variable);
        if *coeff == C::zero() || free_variables.is_empty() {
            Some(*coeff)
        } else {
            None
        }
    }

    fn var_letter() -> char {
        'x'
    }

    fn objective_granularity<C: Coeff>(terms: &[crate::term::Term<C>]) -> Option<C> {
        fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
            a = a.abs();
            b = b.abs();
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            a
        }

        let mut sums: BTreeMap<Vec<usize>, i128> = BTreeMap::new();
        for term in terms {
            if term.indices.is_empty() {
                continue;
            }
            let coeff = term.coeff.to_i128_exact()?;
            let mut key = term.indices.clone();
            key.sort_unstable();
            key.dedup();
            *sums.entry(key).or_insert(0) += coeff;
        }

        let mut granularity = 0i128;
        for (_subset, sum) in sums {
            if sum == 0 {
                continue;
            }
            granularity = gcd_i128(granularity, sum);
        }

        if granularity == 0 {
            None
        } else {
            C::from_i128_checked(granularity)
        }
    }
}

impl VarDomain for Spin {
    const VAR_TYPE: VarType = VarType::Spin;
    type TermStateData = (); // no auxiliary state; term_status holds sign

    fn reduce_indices(indices: &[usize]) -> Vec<usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &i in indices {
            *counts.entry(i).or_insert(0) += 1;
        }
        let mut s: Vec<usize> = counts
            .into_iter()
            .filter(|&(_, cnt)| cnt % 2 == 1)
            .map(|(i, _)| i)
            .collect();
        s.sort_unstable();
        s
    }

    fn fold_term_under_fixes<C: Coeff>(
        term_indices: &[usize],
        coeff: C,
        fixes: &Fixes,
        old_to_new: &[usize],
    ) -> Option<(C, Vec<usize>)> {
        let mut c = coeff;
        for &i in term_indices {
            if fixes.get(i) == Some(false) {
                c = -c;
            }
        }
        let free: Vec<usize> = term_indices
            .iter()
            .filter(|&&i| fixes.get(i).is_none())
            .map(|&i| old_to_new[i])
            .collect();
        Some((c, free))
    }

    fn evaluate_bitsol<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> C {
        let mut value = instance.offset;
        for term in &instance.terms {
            let neg_count = term
                .indices
                .iter()
                .filter(|&&j| !sol.values.contains(j))
                .count();
            if neg_count % 2 == 0 {
                value += term.coeff;
            } else {
                value -= term.coeff;
            }
        }
        value
    }

    fn init_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        sol: &BitSolution,
    ) -> TermState<Self> {
        let mut term_status = Vec::with_capacity(instance.n_terms());
        for term in &instance.terms {
            let neg_count = term
                .indices
                .iter()
                .filter(|&&j| !sol.values.contains(j))
                .count();
            term_status.push(neg_count % 2 == 0);
        }
        TermState {
            term_status,
            data: (),
        }
    }

    fn delta_from_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        _sol: &BitSolution,
        term_state: &TermState<Self>,
    ) -> C {
        let mut delta = C::zero();
        let two = C::from_i64(2);
        for &ti in &instance.var_terms[var] {
            let is_positive = term_state.term_status[ti];
            let term = &instance.terms[ti];
            if is_positive {
                delta -= two * term.coeff;
            } else {
                delta += two * term.coeff;
            }
        }
        delta
    }

    fn flip_with_term_state<C: Coeff>(
        instance: &HuboInstance<C, Self>,
        var: usize,
        sol: &mut BitSolution,
        term_state: &mut TermState<Self>,
    ) {
        sol.flip(var);
        for &ti in &instance.var_terms[var] {
            term_state.term_status[ti] = !term_state.term_status[ti];
        }
    }

    fn local_fields<C: Coeff>(instance: &HuboInstance<C, Self>, sol: &BitSolution) -> Vec<C> {
        let mut h = vec![C::zero(); instance.n_vars()];
        for term in &instance.terms {
            let neg_count = term
                .indices
                .iter()
                .filter(|&&i| !sol.values.contains(i))
                .count();
            let full_prod_pos = neg_count % 2 == 0;
            for &j in &term.indices {
                let sigma_j_pos = sol.values.contains(j);
                if full_prod_pos == sigma_j_pos {
                    h[j] += term.coeff;
                } else {
                    h[j] -= term.coeff;
                }
            }
        }
        h
    }

    fn format_char(high: bool) -> char {
        if high { '+' } else { '-' }
    }

    fn high_to_coeff<C: Coeff>(high: bool) -> C {
        if high { C::one() } else { -C::one() }
    }

    fn is_valid_value<C: Coeff>(value: C) -> bool {
        value == -C::one() || value == C::one()
    }

    fn update_partial_term<C: Coeff>(
        coeff: &mut C,
        free_variables: &mut Vec<usize>,
        variable: usize,
        high: bool,
    ) -> Option<C> {
        if !high {
            *coeff = -*coeff;
        }
        free_variables.retain(|&v| v != variable);
        if *coeff == C::zero() || free_variables.is_empty() {
            Some(*coeff)
        } else {
            None
        }
    }

    fn var_letter() -> char {
        's'
    }

    fn objective_granularity<C: Coeff>(terms: &[crate::term::Term<C>]) -> Option<C> {
        fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
            a = a.abs();
            b = b.abs();
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            a
        }

        let mut subset_sums: BTreeMap<Vec<usize>, i128> = BTreeMap::new();
        for term in terms {
            let coeff = term.coeff.to_i128_exact()?;
            let k = term.indices.len();
            if k == 0 {
                continue;
            }
            let n_masks = 1usize.checked_shl(k as u32)?;
            for mask in 1..n_masks {
                let mut subset = Vec::with_capacity(mask.count_ones() as usize);
                for (pos, &idx) in term.indices.iter().enumerate() {
                    if (mask & (1usize << pos)) != 0 {
                        subset.push(idx);
                    }
                }
                *subset_sums.entry(subset).or_insert(0) += coeff;
            }
        }

        let mut granularity = 0i128;
        for (subset, sum) in subset_sums {
            if sum == 0 {
                continue;
            }
            let factor = 1i128.checked_shl(subset.len() as u32)?;
            let value = sum.checked_mul(factor)?;
            granularity = gcd_i128(granularity, value);
        }

        if granularity == 0 {
            None
        } else {
            C::from_i128_checked(granularity)
        }
    }

    fn objective_grid_base<C: Coeff>(offset: C, terms: &[crate::term::Term<C>]) -> C {
        let mut value = offset;
        for term in terms {
            if term.indices.len() % 2 == 0 {
                value += term.coeff;
            } else {
                value -= term.coeff;
            }
        }
        value
    }
}
