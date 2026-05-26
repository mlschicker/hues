//! Compact bit-packed HUBO solutions.
//!
//! [`BitSolution`] stores a complete variable assignment as a single
//! [`FixedBitSet`] — bit `i` is set iff variable `i` holds its "high" value
//! (BIN = 1, SPIN = +1).  This eliminates the need to store solutions as
//! `Vec<C>` and makes objective evaluation product-free (parity/subset
//! checks instead of repeated multiplication).

use fixedbitset::FixedBitSet;

use crate::coeff::Coeff;
use crate::domain::{Bin, Spin, VarDomain, VarType};
use crate::instance::HuboInstance;

/// A complete HUBO solution stored as a bit vector.
///
/// Bit `i` is set iff variable `i` holds its "high" value:
/// - **BIN**: bit set → variable = 1
/// - **SPIN**: bit set → variable = +1
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSolution {
    pub values: FixedBitSet,
}

impl BitSolution {
    /// Create an all-zero (all-low) solution for `n_vars` variables.
    pub fn new(n_vars: usize) -> Self {
        Self {
            values: FixedBitSet::with_capacity(n_vars),
        }
    }

    pub fn from_bool_vec(v: Vec<bool>) -> Self {
        let mut bs = Self::new(v.len());
        for (i, val) in v.into_iter().enumerate() {
            bs.values.set(i, val);
        }
        bs
    }

    /// Build a `BitSolution` from a domain-value slice.
    ///
    /// For both BIN and SPIN, `C::ONE` maps to the "high" bit (1 / +1).
    pub fn from_vec<C: Coeff>(v: &[C]) -> Self {
        let mut bs = Self::new(v.len());
        for (i, &val) in v.iter().enumerate() {
            bs.values.set(i, val == C::one());
        }
        bs
    }

    pub fn get(&self, var_idx: usize) -> bool {
        self.values.contains(var_idx)
    }

    /// Convert to a domain-value `Vec<C>` (for use at system boundaries such
    /// as kernelization lifting or file I/O).
    pub fn to_vec<C: Coeff>(&self, var_type: VarType) -> Vec<C> {
        match var_type {
            VarType::Bin => self.to_vec_typed::<C, Bin>(),
            VarType::Spin => self.to_vec_typed::<C, Spin>(),
        }
    }

    pub fn to_vec_typed<C: Coeff, V: VarDomain>(&self) -> Vec<C> {
        (0..self.values.len())
            .map(|i| V::high_to_coeff::<C>(self.values.contains(i)))
            .collect()
    }

    /// Flip variable `var` in place.
    #[inline]
    pub fn flip(&mut self, var: usize) {
        self.values.toggle(var);
    }

    /// Compute every local field from scratch.
    ///
    /// h[j] = -∂E/∂σ_j = Σ_{T ∋ j}  c_T · ∏_{i ∈ T\{j}} σ_i
    ///
    /// Equivalently:  E changes by  ΔE = -2 σ_j h_j  when σ_j flips.
    pub fn local_fields<C: Coeff, V: VarDomain>(&self, instance: &HuboInstance<C, V>) -> Vec<C> {
        V::local_fields(instance, self)
    }

    /// Evaluate the objective value for this solution.
    pub fn evaluate<C: Coeff, V: VarDomain>(&self, instance: &HuboInstance<C, V>) -> C {
        instance.evaluate_bitsol(self)
    }

    /// Format the solution as a compact human-readable string.
    ///
    /// BIN: `"101"` (each character is '0' or '1')
    /// SPIN: `"+-+"` (each character is '+' or '-')
    pub fn format_string(&self, var_type: VarType) -> String {
        match var_type {
            VarType::Bin => self.format_string_typed::<Bin>(),
            VarType::Spin => self.format_string_typed::<Spin>(),
        }
    }

    pub fn format_string_typed<V: VarDomain>(&self) -> String {
        (0..self.values.len())
            .map(|i| V::format_char(self.values.contains(i)))
            .collect()
    }

    /// Return the solution as a JSON-ready `Vec<i64>` of domain values.
    ///
    /// BIN: 0 or 1.  SPIN: -1 or +1.
    pub fn to_json_array(&self, var_type: VarType) -> Vec<i64> {
        (0..self.values.len())
            .map(|i| match var_type {
                VarType::Bin => if self.values.contains(i) { 1 } else { 0 },
                VarType::Spin => if self.values.contains(i) { 1 } else { -1 },
            })
            .collect()
    }

    /// Write the solution in HUES solution-file format.
    ///
    /// Emits lines of the form `x0 = 1` (BIN) or `s0 = -1` (SPIN) to `f`.
    pub fn write_to<W: std::io::Write>(&self, f: &mut W, var_type: VarType) -> std::io::Result<()> {
        match var_type {
            VarType::Bin => self.write_to_typed::<W, Bin>(f),
            VarType::Spin => self.write_to_typed::<W, Spin>(f),
        }
    }

    pub fn write_to_typed<W: std::io::Write, V: VarDomain>(
        &self,
        f: &mut W,
    ) -> std::io::Result<()> {
        let letter = V::var_letter();
        for i in 0..self.values.len() {
            let high = self.values.contains(i);
            let val = match V::VAR_TYPE {
                VarType::Bin => {
                    if high {
                        "1"
                    } else {
                        "0"
                    }
                }
                VarType::Spin => {
                    if high {
                        "1"
                    } else {
                        "-1"
                    }
                }
            };
            writeln!(f, "{letter}{i} = {val}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HuboModel;

    #[test]
    fn binary_delta_respects_current_bit_of_flipped_var() {
        let instance = HuboModel::binary(2).add_term(&[0, 1], 5.0).build();
        let sol = BitSolution::from_vec(&[1.0, 0.0]);
        let term_state = instance.init_term_state(&sol);
        let delta_cache = instance.init_delta_cache(&sol, &term_state);

        assert!((delta_cache.deltas[0]).abs() < 1e-12);
        assert!((delta_cache.deltas[1] - 5.0).abs() < 1e-12);
    }
}
