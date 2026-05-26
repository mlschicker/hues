//! Fluent builder API for constructing [`HuboInstance`]s programmatically.
//!
//! # Example
//!
//! ```rust
//! use hues::model::HuboModel;
//!
//! // min  2 * x0 * x1  -  3 * x2  +  1.5
//! let instance = HuboModel::binary(3)
//!     .with_offset(1.5)
//!     .add_term(&[0, 1], 2.0)
//!     .add_term(&[2], -3.0)
//!     .build();
//! ```

use std::collections::HashMap;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::coeff::Coeff;
use crate::{domain::{Bin, Spin, VarDomain, VarType}, instance::HuboInstance, term::Term};

// ---------------------------------------------------------------------------
// Polynomial expression DSL
// ---------------------------------------------------------------------------

/// A symbolic polynomial expression over HUBO variables.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr<C: Coeff> {
    offset: C,
    terms: FxHashMap<Vec<usize>, C>,
}

impl<C: Coeff> Expr<C> {
    pub fn zero() -> Self {
        Self {
            offset: C::zero(),
            terms: FxHashMap::default(),
        }
    }

    pub fn constant(value: C) -> Self {
        Self {
            offset: value,
            terms: FxHashMap::default(),
        }
    }

    pub fn var(var: usize) -> Self {
        let mut terms = FxHashMap::default();
        terms.insert(vec![var], C::one());
        Self {
            offset: C::zero(),
            terms,
        }
    }

    pub fn with_term(mut self, indices: Vec<usize>, coeff: C) -> Self {
        self.add_term_mut(indices, coeff);
        self
    }

    pub fn add_term_mut(&mut self, mut indices: Vec<usize>, coeff: C) {
        if coeff == C::zero() {
            return;
        }
        indices.sort_unstable();
        self.add_sorted_term_mut(indices, coeff);
    }

    fn add_sorted_term_mut(&mut self, indices: Vec<usize>, coeff: C) {
        if coeff == C::zero() {
            return;
        }

        if indices.is_empty() {
            self.offset += coeff;
            return;
        }

        match self.terms.entry(indices) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(coeff);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let value = entry.get_mut();
                *value += coeff;
                if *value == C::zero() {
                    entry.remove_entry();
                }
            }
        }
    }

    fn merge_sorted_indices(lhs_idx: &[usize], rhs_idx: &[usize]) -> Vec<usize> {
        let mut merged = Vec::with_capacity(lhs_idx.len() + rhs_idx.len());

        let mut i = 0;
        let mut j = 0;

        while i < lhs_idx.len() && j < rhs_idx.len() {
            if lhs_idx[i] <= rhs_idx[j] {
                merged.push(lhs_idx[i]);
                i += 1;
            } else {
                merged.push(rhs_idx[j]);
                j += 1;
            }
        }

        if i < lhs_idx.len() {
            merged.extend_from_slice(&lhs_idx[i..]);
        }
        if j < rhs_idx.len() {
            merged.extend_from_slice(&rhs_idx[j..]);
        }

        merged
    }

    pub fn pow(&self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::constant(C::one());
        }

        if exp == 1 {
            return self.clone();
        }

        if exp == 2 {
            return self.square();
        }

        let mut result = Self::constant(C::one());
        let mut base = self.clone();

        while exp > 0 {
            if exp & 1 == 1 {
                result *= base.clone();
            }
            exp >>= 1;
            if exp > 0 {
                base = base.clone() * base;
            }
        }
        result
    }

    fn square(&self) -> Self {
        if self.terms.is_empty() {
            return Self::constant(self.offset * self.offset);
        }

        let n_terms = self.terms.len();
        let pair_count = n_terms.saturating_mul(n_terms.saturating_add(1)) / 2;

        let mut out = Expr {
            offset: self.offset * self.offset,
            terms: FxHashMap::with_capacity_and_hasher(
                n_terms.saturating_mul(n_terms).saturating_add(n_terms),
                Default::default(),
            ),
        };

        if self.offset != C::zero() {
            let two = C::one() + C::one();
            for (indices, coeff) in &self.terms {
                out.add_sorted_term_mut(indices.clone(), (*coeff * self.offset) * two);
            }
        }

        let entries: Vec<(&Vec<usize>, C)> = self
            .terms
            .iter()
            .map(|(idx, coeff)| (idx, *coeff))
            .collect();

        const SORT_AGGREGATE_THRESHOLD: usize = 2048;
        if pair_count >= SORT_AGGREGATE_THRESHOLD {
            let mut products: Vec<(Vec<usize>, C)> = Vec::with_capacity(pair_count);

            for i in 0..entries.len() {
                for j in i..entries.len() {
                    let mut coeff = entries[i].1 * entries[j].1;
                    if i != j {
                        coeff += coeff;
                    }
                    if coeff != C::zero() {
                        let merged = Self::merge_sorted_indices(entries[i].0, entries[j].0);
                        products.push((merged, coeff));
                    }
                }
            }

            products.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let mut iter = products.into_iter();
            if let Some((mut current_indices, mut current_coeff)) = iter.next() {
                for (indices, coeff) in iter {
                    if indices == current_indices {
                        current_coeff += coeff;
                    } else {
                        out.add_sorted_term_mut(current_indices, current_coeff);
                        current_indices = indices;
                        current_coeff = coeff;
                    }
                }
                out.add_sorted_term_mut(current_indices, current_coeff);
            }
        } else {
            for i in 0..entries.len() {
                for j in i..entries.len() {
                    let mut coeff = entries[i].1 * entries[j].1;
                    if i != j {
                        coeff += coeff;
                    }
                    let merged = Self::merge_sorted_indices(entries[i].0, entries[j].0);
                    out.add_sorted_term_mut(merged, coeff);
                }
            }
        }

        out
    }
}

impl<C: Coeff> Default for Expr<C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<C: Coeff> From<C> for Expr<C> {
    fn from(value: C) -> Self {
        Self::constant(value)
    }
}

impl<C: Coeff> AddAssign for Expr<C> {
    fn add_assign(&mut self, rhs: Self) {
        self.offset += rhs.offset;
        for (indices, coeff) in rhs.terms {
            self.add_term_mut(indices, coeff);
        }
    }
}

impl<C: Coeff> Add for Expr<C> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<C: Coeff> SubAssign for Expr<C> {
    fn sub_assign(&mut self, rhs: Self) {
        self.offset -= rhs.offset;
        for (indices, coeff) in rhs.terms {
            self.add_term_mut(indices, -coeff);
        }
    }
}

impl<C: Coeff> Sub for Expr<C> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<C: Coeff> Neg for Expr<C> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.offset = -self.offset;
        for coeff in self.terms.values_mut() {
            *coeff = -*coeff;
        }
        self
    }
}

impl<C: Coeff> MulAssign for Expr<C> {
    fn mul_assign(&mut self, rhs: Self) {
        let lhs = std::mem::take(self);
        *self = lhs * rhs;
    }
}

impl<C: Coeff> Mul for Expr<C> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_offset = self.offset;
        let rhs_offset = rhs.offset;
        let lhs_terms = self.terms;
        let rhs_terms = rhs.terms;

        if lhs_terms.is_empty() || rhs_terms.is_empty() {
            let mut out = Expr::zero();
            out.offset = lhs_offset * rhs_offset;

            if rhs_offset != C::zero() {
                for (indices, coeff) in lhs_terms {
                    out.add_sorted_term_mut(indices, coeff * rhs_offset);
                }
            }

            if lhs_offset != C::zero() {
                for (indices, coeff) in rhs_terms {
                    out.add_sorted_term_mut(indices, coeff * lhs_offset);
                }
            }

            return out;
        }

        let estimated_terms = lhs_terms
            .len()
            .saturating_mul(rhs_terms.len())
            .saturating_add(lhs_terms.len())
            .saturating_add(rhs_terms.len());
        let cross_terms = lhs_terms.len().saturating_mul(rhs_terms.len());

        let mut out = Expr {
            offset: lhs_offset * rhs_offset,
            terms: FxHashMap::with_capacity_and_hasher(estimated_terms, Default::default()),
        };

        if rhs_offset != C::zero() {
            for (indices, coeff) in &lhs_terms {
                out.add_sorted_term_mut(indices.clone(), *coeff * rhs_offset);
            }
        }

        if lhs_offset != C::zero() {
            for (indices, coeff) in &rhs_terms {
                out.add_sorted_term_mut(indices.clone(), *coeff * lhs_offset);
            }
        }

        const SORT_AGGREGATE_THRESHOLD: usize = 2048;
        if cross_terms >= SORT_AGGREGATE_THRESHOLD {
            let mut products: Vec<(Vec<usize>, C)> = Vec::with_capacity(cross_terms);

            for (lhs_idx, lhs_coeff) in &lhs_terms {
                for (rhs_idx, rhs_coeff) in &rhs_terms {
                    let coeff = *lhs_coeff * *rhs_coeff;
                    if coeff != C::zero() {
                        let merged = Self::merge_sorted_indices(lhs_idx, rhs_idx);
                        products.push((merged, coeff));
                    }
                }
            }

            products.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let mut iter = products.into_iter();
            if let Some((mut current_indices, mut current_coeff)) = iter.next() {
                for (indices, coeff) in iter {
                    if indices == current_indices {
                        current_coeff += coeff;
                    } else {
                        out.add_sorted_term_mut(current_indices, current_coeff);
                        current_indices = indices;
                        current_coeff = coeff;
                    }
                }

                out.add_sorted_term_mut(current_indices, current_coeff);
            }
        } else {
            for (lhs_idx, lhs_coeff) in &lhs_terms {
                for (rhs_idx, rhs_coeff) in &rhs_terms {
                    let merged = Self::merge_sorted_indices(lhs_idx, rhs_idx);
                    out.add_sorted_term_mut(merged, *lhs_coeff * *rhs_coeff);
                }
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A fluent builder for [`HuboInstance`].
pub struct HuboModel<C: Coeff, V: VarDomain> {
    n_vars: usize,
    offset: C,
    metadata: Vec<(String, String)>,
    terms: Vec<Term<C>>,
    _marker: PhantomData<fn() -> V>,
}

impl<C: Coeff> HuboModel<C, Bin> {
    /// Create a new model with binary variables (x_i in {0, 1}).
    pub fn binary(n_vars: usize) -> Self {
        Self::new(n_vars)
    }

    /// Create a binary-variable logical negation expression: `1 - x_var`.
    pub fn expr_not_var(&self, var: usize) -> Expr<C> {
        assert!(var < self.n_vars, "index {var} out of range for n_vars = {}", self.n_vars);
        Expr::constant(C::one()) - self.expr_var(var)
    }
}

impl<C: Coeff> HuboModel<C, Spin> {
    /// Create a new model with spin variables (s_i in {-1, +1}).
    pub fn spin(n_vars: usize) -> Self {
        Self::new(n_vars)
    }
}

impl<C: Coeff, V: VarDomain> HuboModel<C, V> {
    fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            offset: C::zero(),
            metadata: Vec::new(),
            terms: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn with_offset(mut self, offset: C) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_index_base(self) -> Self {
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    pub fn expr_var(&self, var: usize) -> Expr<C> {
        assert!(var < self.n_vars, "index {var} out of range for n_vars = {}", self.n_vars);
        Expr::var(var)
    }

    pub fn expr_const(&self, value: C) -> Expr<C> {
        Expr::constant(value)
    }

    pub fn expr_neg_var(&self, var: usize) -> Expr<C> {
        -self.expr_var(var)
    }

    pub fn add_term(mut self, indices: &[usize], coeff: C) -> Self {
        self.push_term(indices, coeff);
        self
    }

    pub fn add_term_mut(&mut self, indices: &[usize], coeff: C) -> &mut Self {
        self.push_term(indices, coeff);
        self
    }

    pub fn add_linear(self, var: usize, coeff: C) -> Self {
        self.add_term(&[var], coeff)
    }

    pub fn add_quadratic(self, var_i: usize, var_j: usize, coeff: C) -> Self {
        self.add_term(&[var_i, var_j], coeff)
    }

    pub fn add_constant(mut self, value: C) -> Self {
        self.offset += value;
        self
    }

    pub fn add_expr(mut self, expr: Expr<C>) -> Self {
        self.push_expr(expr);
        self
    }

    pub fn add_expr_mut(&mut self, expr: Expr<C>) -> &mut Self {
        self.push_expr(expr);
        self
    }

    pub fn add_terms(mut self, terms: impl IntoIterator<Item = (Vec<usize>, C)>) -> Self {
        for (indices, coeff) in terms {
            self.push_term(&indices, coeff);
        }
        self
    }

    pub fn build(self) -> HuboInstance<C, V> {
        HuboInstance::new(self.n_vars, self.offset, self.terms)
    }

    fn push_term(&mut self, indices: &[usize], coeff: C) {
        for &idx in indices {
            assert!(idx < self.n_vars, "index {idx} out of range for n_vars = {}", self.n_vars);
        }

        let sorted = V::reduce_indices(indices);

        if sorted.is_empty() {
            self.offset += coeff;
        } else {
            self.terms.push(Term { indices: sorted, coeff });
        }
    }

    fn push_expr(&mut self, expr: Expr<C>) {
        self.offset += expr.offset;

        let mut reduced_terms: HashMap<Vec<usize>, C> = HashMap::new();

        for (indices, coeff) in expr.terms {
            for &idx in &indices {
                assert!(idx < self.n_vars, "index {idx} out of range for n_vars = {}", self.n_vars);
            }

            let reduced = V::reduce_indices(&indices);
            if reduced.is_empty() {
                self.offset += coeff;
            } else {
                let entry = reduced_terms.entry(reduced.clone()).or_insert(C::zero());
                *entry += coeff;
                if *entry == C::zero() {
                    reduced_terms.remove(&reduced);
                }
            }
        }

        for (indices, coeff) in reduced_terms {
            self.terms.push(Term { indices, coeff });
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: From<HuboModel> for HuboInstance
// ---------------------------------------------------------------------------

impl<C: Coeff, V: VarDomain> From<HuboModel<C, V>> for HuboInstance<C, V> {
    fn from(builder: HuboModel<C, V>) -> Self {
        builder.build()
    }
}

// ---------------------------------------------------------------------------
// Convenience: serialise an instance back to HUBO-TL text
// ---------------------------------------------------------------------------

impl<C: Coeff, V: VarDomain> HuboInstance<C, V> {
    fn to_f64_clone<V2: VarDomain>(&self) -> HuboInstance<f64, V2> {
        let terms: Vec<Term<f64>> = self
            .terms
            .iter()
            .map(|t| Term {
                indices: t.indices.clone(),
                coeff: t.coeff.to_f64(),
            })
            .collect();
        HuboInstance::new(self.n_vars(), self.offset.to_f64(), terms)
    }

    fn merge_term(map: &mut HashMap<Vec<usize>, f64>, indices: Vec<usize>, coeff: f64) {
        if coeff == 0.0 {
            return;
        }
        let entry = map.entry(indices.clone()).or_insert(0.0);
        *entry += coeff;
        if *entry == 0.0 {
            map.remove(&indices);
        }
    }

    fn for_each_subset(indices: &[usize], mut f: impl FnMut(&[usize])) {
        fn rec(
            indices: &[usize],
            pos: usize,
            subset: &mut Vec<usize>,
            f: &mut impl FnMut(&[usize]),
        ) {
            if pos == indices.len() {
                f(subset);
                return;
            }

            rec(indices, pos + 1, subset, f);
            subset.push(indices[pos]);
            rec(indices, pos + 1, subset, f);
            subset.pop();
        }

        let mut subset = Vec::new();
        rec(indices, 0, &mut subset, &mut f);
    }
}

impl<C: Coeff> HuboInstance<C, Bin> {
    /// Convert to an equivalent HUBO (binary) instance.
    pub fn to_hubo(&self) -> HuboInstance<f64, Bin> {
        self.to_f64_clone::<Bin>()
    }

    /// Convert to an equivalent HUSO (spin) instance.
    pub fn to_huso(&self) -> HuboInstance<f64, Spin> {
        // x_i = (s_i + 1) / 2  → expand each monomial.
        let mut offset = self.offset.to_f64();
        let mut merged: HashMap<Vec<usize>, f64> = HashMap::new();

        for term in &self.terms {
            let k = term.indices.len();
            let coeff = term.coeff.to_f64();
            let factor = coeff / 2f64.powi(k as i32);

            Self::for_each_subset(&term.indices, |subset| {
                if subset.is_empty() {
                    offset += factor;
                } else {
                    Self::merge_term(&mut merged, subset.to_vec(), factor);
                }
            });
        }

        let mut terms: Vec<Term<f64>> = merged
            .into_iter()
            .map(|(indices, coeff)| Term { indices, coeff })
            .collect();
        terms.sort_by(|a, b| a.indices.cmp(&b.indices));

        HuboInstance::new(self.n_vars(), offset, terms)
    }
}

impl<C: Coeff> HuboInstance<C, Spin> {
    /// Convert to an equivalent HUBO (binary) instance.
    pub fn to_hubo(&self) -> HuboInstance<f64, Bin> {
        // s_i = 2x_i - 1  → expand each monomial.
        let mut offset = self.offset.to_f64();
        let mut merged: HashMap<Vec<usize>, f64> = HashMap::new();

        for term in &self.terms {
            let k = term.indices.len();
            let coeff = term.coeff.to_f64();

            Self::for_each_subset(&term.indices, |subset| {
                let subset_size = subset.len();
                let sign = if (k - subset_size).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let factor = sign * 2f64.powi(subset_size as i32);
                let sub_coeff = coeff * factor;

                if subset.is_empty() {
                    offset += sub_coeff;
                } else {
                    Self::merge_term(&mut merged, subset.to_vec(), sub_coeff);
                }
            });
        }

        let mut terms: Vec<Term<f64>> = merged
            .into_iter()
            .map(|(indices, coeff)| Term { indices, coeff })
            .collect();
        terms.sort_by(|a, b| a.indices.cmp(&b.indices));

        HuboInstance::new(self.n_vars(), offset, terms)
    }

    pub fn to_huso(&self) -> HuboInstance<f64, Spin> {
        self.to_f64_clone::<Spin>()
    }
}

impl<C: Coeff, V: VarDomain> HuboInstance<C, V> {
    /// Serialise the instance to the HUBO-TL text format.
    pub fn to_hubo_tl(&self, metadata: Option<Vec<(String, String)>>) -> String {
        self.to_text_with_magic("HUBO", metadata.unwrap_or_default())
    }

    /// Serialise the instance to the HUSO-TL text format.
    pub fn to_huso_tl(&self, metadata: Option<Vec<(String, String)>>) -> String {
        self.to_text_with_magic("HUSO", metadata.unwrap_or_default())
    }

    /// Serialise the instance to JSON.
    ///
    /// Produces a structured object compatible with `parse_auto` / the solver's
    /// `--convert-to json` mode.  Integer-valued coefficients are written as
    /// JSON integers; fractional ones as floats.
    pub fn to_json(&self, metadata: Option<Vec<(String, String)>>) -> String {
        fn coeff_val<C: Coeff>(c: C) -> serde_json::Value {
            let f = c.to_f64();
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.007_199_254_740_992e15 {
                serde_json::Value::Number(serde_json::Number::from(f as i64))
            } else {
                serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }

        let var_type_str = match V::VAR_TYPE {
            VarType::Bin => "BIN",
            VarType::Spin => "SPIN",
        };

        let terms: Vec<serde_json::Value> = self.terms.iter().map(|t| {
            serde_json::json!({
                "indices": t.indices,
                "coeff": coeff_val(t.coeff),
            })
        }).collect();

        let mut obj = serde_json::json!({
            "var_type": var_type_str,
            "n_vars": self.n_vars(),
            "offset": coeff_val(self.offset),
            "terms": terms,
        });

        if let Some(meta) = metadata
            && !meta.is_empty() {
                let map: serde_json::Map<String, serde_json::Value> = meta
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect();
                obj["metadata"] = serde_json::Value::Object(map);
            }

        serde_json::to_string_pretty(&obj).unwrap()
    }

    fn to_text_with_magic(&self, magic: &str, metadata: Vec<(String, String)>) -> String {
        let mut out = String::new();

        out.push_str(&format!("{magic} 1\n"));

        for (key, value) in metadata {
            out.push_str(&format!("META {key}={value}\n"));
        }

        let var_str = match V::VAR_TYPE {
            VarType::Bin => "BIN",
            VarType::Spin => "SPIN",
        };
        out.push_str(&format!("VAR_TYPE {var_str}\n"));
        out.push_str(&format!("N {}\n", self.n_vars()));
        out.push_str(&format!("M {}\n", self.terms.len()));

        if self.offset != C::zero() {
            out.push_str(&format!("OFFSET {}\n", self.offset));
        }

        let mut sorted_terms: Vec<&Term<C>> = self.terms.iter().collect();
        sorted_terms.sort_by(|a, b| {
            a.indices
                .cmp(&b.indices)
                .then_with(|| a.coeff.to_string().cmp(&b.coeff.to_string()))
        });

        for term in sorted_terms {
            let indices: Vec<String> = term.indices.iter().map(|&i| i.to_string()).collect();
            out.push_str(&format!("{} {}\n", indices.join(" "), term.coeff));
        }

        out
    }

    /// Write the instance to a file in HUBO-TL format.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_hubo_tl(None))
    }

    /// Evaluate the objective value for a given solution in domain values.
    ///
    /// `solution` must contain values in the original domain:
    /// - **BIN**: 0 or 1
    /// - **SPIN**: -1 or +1
    pub fn evaluate(&self, solution: &[C]) -> C {
        assert_eq!(
            solution.len(),
            self.n_vars(),
            "solution length {} does not match n_vars {}",
            solution.len(),
            self.n_vars(),
        );

        let mut value = self.offset;
        for term in &self.terms {
            let product: C = term.indices.iter().map(|&i| solution[i]).product();
            value += term.coeff * product;
        }
        value
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    #[test]
    fn build_binary_instance() {
        let instance = HuboModel::binary(3)
            .with_offset(1.5)
            .add_term(&[0, 1], 2.0)
            .add_term(&[2], -3.0)
            .build();

        assert_eq!(instance.var_type(), VarType::Bin);
        assert_eq!(instance.n_vars(), 3);
        assert_eq!(instance.n_terms(), 2);
        assert_eq!(instance.offset, 1.5);
    }

    #[test]
    fn build_spin_instance() {
        let instance = HuboModel::spin(4)
            .add_quadratic(0, 1, -1.0)
            .add_quadratic(2, 3, -1.0)
            .add_linear(0, 0.5)
            .build();

        assert_eq!(instance.var_type(), VarType::Spin);
        assert_eq!(instance.n_vars(), 4);
        assert_eq!(instance.n_terms(), 3);
    }

    #[test]
    fn deduplicates_and_sorts_indices() {
        let instance = HuboModel::binary(5).add_term(&[3, 1, 3, 1, 2], 1.0).build();
        assert_eq!(instance.terms[0].indices, vec![1, 2, 3]);
    }

    #[test]
    fn spin_pairwise_reduction() {
        let instance = HuboModel::spin(3).add_term(&[0, 0], 4.0).build();
        assert_eq!(instance.terms.len(), 0);
        assert!((instance.offset - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn spin_pairwise_odd_keeps_one() {
        let instance = HuboModel::spin(3).add_term(&[1, 1, 1, 2], 2.5).build();
        assert_eq!(instance.terms.len(), 1);
        assert_eq!(instance.terms[0].indices, vec![1, 2]);
    }

    #[test]
    fn spin_mixed_pairwise() {
        let instance = HuboModel::spin(3).add_term(&[0, 0, 1, 1, 2], 7.0).build();
        assert_eq!(instance.terms.len(), 1);
        assert_eq!(instance.terms[0].indices, vec![2]);
    }

    #[test]
    fn add_term_mut_in_loop() {
        let mut model = HuboModel::binary(3);
        for i in 0..3 {
            model.add_term_mut(&[i], i as f64);
        }
        let instance = model.build();
        assert_eq!(instance.n_terms(), 3);
    }

    #[test]
    fn add_terms_bulk() {
        let instance = HuboModel::binary(4)
            .add_terms(vec![(vec![0, 1], 1.0), (vec![2, 3], 2.0), (vec![0], -0.5)])
            .build();
        assert_eq!(instance.n_terms(), 3);
    }

    #[test]
    fn add_constant() {
        let instance = HuboModel::binary(2)
            .with_offset(1.0)
            .add_constant(2.5)
            .build();
        assert_eq!(instance.offset, 3.5);
    }

    #[test]
    fn roundtrip_via_hubo_tl() {
        let original = HuboModel::binary(3)
            .with_offset(0.5)
            .with_meta("author", "test")
            .add_term(&[0, 1], 2.0)
            .add_term(&[2], -3.0)
            .build();

        let text = original.to_hubo_tl(None);
        let parsed = parser::parse::<f64>(&text).expect("roundtrip parse failed");
        let parsed_inst = match parsed.0 {
            crate::instance::HuboInstanceEnum::Bin(i) => i,
            _ => panic!("expected BIN"),
        };
        assert_eq!(parsed_inst.offset, original.offset);
        assert_eq!(parsed_inst.terms, original.terms);
    }

    #[test]
    fn roundtrip_spin() {
        let original = HuboModel::spin(2).add_quadratic(0, 1, -1.0).build();

        let text = original.to_hubo_tl(None);
        let parsed = parser::parse::<f64>(&text).expect("roundtrip parse failed");
        let parsed_inst = match parsed.0 {
            crate::instance::HuboInstanceEnum::Spin(i) => i,
            _ => panic!("expected SPIN"),
        };
        assert_eq!(parsed_inst.terms, original.terms);
    }

    #[test]
    fn from_trait() {
        let model = HuboModel::binary(2).add_linear(0, 1.0);
        let instance: HuboInstance<f64, Bin> = model.into();
        assert_eq!(instance.n_terms(), 1);
    }

    #[test]
    #[should_panic(expected = "index 5 out of range")]
    fn panics_on_out_of_range_index() {
        HuboModel::binary(3).add_term(&[5], 1.0);
    }

    #[test]
    fn evaluate_binary() {
        let instance = HuboModel::binary(3)
            .with_offset(1.5)
            .add_term(&[0, 1], 2.0)
            .add_term(&[2], -3.0)
            .build();

        assert!((instance.evaluate(&[1.0, 1.0, 0.0]) - 3.5).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[0.0, 0.0, 1.0]) - (-1.5)).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[1.0, 0.0, 1.0]) - (-1.5)).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[0.0, 0.0, 0.0]) - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_spin() {
        let instance = HuboModel::spin(4)
            .add_quadratic(0, 1, -1.0)
            .add_quadratic(2, 3, -1.0)
            .build();

        assert!((instance.evaluate(&[1.0, 1.0, 1.0, 1.0]) - (-2.0)).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[1.0, -1.0, 1.0, -1.0]) - 2.0).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[1.0, 1.0, -1.0, -1.0]) - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_higher_order() {
        let instance = HuboModel::binary(3).add_term(&[0, 1, 2], 5.0).build();
        assert!((instance.evaluate(&[1.0, 1.0, 1.0]) - 5.0).abs() < f64::EPSILON);
        assert!((instance.evaluate(&[1.0, 1.0, 0.0]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "solution length 2 does not match n_vars 3")]
    fn evaluate_wrong_length_panics() {
        let instance = HuboModel::binary(3).add_linear(0, 1.0).build();
        instance.evaluate(&[1.0, 0.0]);
    }
}
