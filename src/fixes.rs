use core::fmt;

use fixedbitset::FixedBitSet;

use crate::{
    Coeff,
    domain::{VarDomain, VarType},
};

/// Represents the variable fixings at a node in the search tree.
#[derive(Debug, Clone)]
pub struct Fixes {
    /// `assigned[i]` is set iff variable `i` has been fixed at this node
    pub assigned: FixedBitSet,
    /// `values[i]` encodes the "high" flag: BIN `true` -> 1, SPIN `true` -> +1.
    pub values: FixedBitSet,
}

impl Fixes {
    pub fn new(n_vars: usize) -> Self {
        Self {
            assigned: FixedBitSet::with_capacity(n_vars),
            values: FixedBitSet::with_capacity(n_vars),
        }
    }

    pub fn fix_variable<C: Coeff, V: VarDomain>(
        &mut self,
        var_idx: usize,
        value: C,
    ) -> Result<(), FixError> {
        let n_vars = self.assigned.len();

        if var_idx >= n_vars {
            return Err(FixError::InvalidVariableIndex {
                index: var_idx,
                n_vars,
            });
        }

        if !V::is_valid_value::<C>(value) {
            return Err(FixError::InvalidFixValue {
                index: var_idx,
                value: value.to_string(),
                var_type: V::VAR_TYPE,
            });
        }

        if self.assigned.contains(var_idx) && (self.values.contains(var_idx) != (value == C::one()))
        {
            return Err(FixError::ConflictingFixes { index: var_idx });
        } else {
            self.assigned.insert(var_idx);
            self.values.set(var_idx, value == C::one());
        }

        Ok(())
    }

    pub fn set(&mut self, var_idx: usize, high: bool) -> Result<(), FixError> {
        let n_vars = self.assigned.len();

        if var_idx >= n_vars {
            return Err(FixError::InvalidVariableIndex {
                index: var_idx,
                n_vars,
            });
        }

        if self.assigned.contains(var_idx) && (self.values.contains(var_idx) != high) {
            return Err(FixError::ConflictingFixes { index: var_idx });
        } else {
            self.assigned.insert(var_idx);
            self.values.set(var_idx, high);
        }

        Ok(())
    }

    /// Compose two fixes together.
    pub fn compose(&self, step: &Fixes) -> Result<Self, FixError> {
        let mut assigned = self.assigned.clone();
        let mut values = self.values.clone();

        for step_idx in step.assigned.ones() {
            if assigned.contains(step_idx)
                && (values.contains(step_idx) != step.values.contains(step_idx))
            {
                return Err(FixError::ConflictingFixes { index: step_idx });
            } else {
                assigned.insert(step_idx);
                values.set(step_idx, step.values.contains(step_idx));
            }
        }
        Ok(Self { assigned, values })
    }

    pub fn clear(&mut self) {
        self.assigned.clear();
        self.values.clear();
    }

    pub fn num_fixed(&self) -> usize {
        self.assigned.count_ones(..)
    }

    pub fn num_free(&self) -> usize {
        self.assigned.len() - self.assigned.count_ones(..)
    }

    pub fn get(&self, var_idx: usize) -> Option<bool> {
        if self.assigned.contains(var_idx) {
            Some(self.values.contains(var_idx))
        } else {
            None
        }
    }

    pub fn iter_fixed(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        self.assigned
            .ones()
            .map(move |idx| (idx, self.values.contains(idx)))
    }

    pub fn iter_free(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.assigned.len()).filter(move |&idx| !self.assigned.contains(idx))
    }

    pub fn iter_unassigned(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.assigned.len()).filter(move |&idx| !self.assigned.contains(idx))
    }
}

/// Check whether a given fixed value is valid for the variable type.
pub fn is_valid_value<C: Coeff>(var_type: VarType, value: C) -> bool {
    match var_type {
        VarType::Bin => value == C::zero() || value == C::one(),
        VarType::Spin => value == -C::one() || value == C::one(),
    }
}

#[derive(Debug, Clone)]
pub enum FixError {
    ConflictingFixes {
        index: usize,
    },
    InvalidVariableIndex {
        index: usize,
        n_vars: usize,
    },
    InvalidFixValue {
        index: usize,
        value: String,
        var_type: VarType,
    },
}

impl fmt::Display for FixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixError::ConflictingFixes { index } => {
                write!(f, "Conflicting fixes for variable index {}", index)
            }
            FixError::InvalidVariableIndex { index, n_vars } => {
                write!(
                    f,
                    "Variable index {} is out of bounds (0..{})",
                    index, n_vars
                )
            }
            FixError::InvalidFixValue {
                index,
                value,
                var_type,
            } => {
                write!(
                    f,
                    "Invalid fix value {} for variable index {} (type: {:?})",
                    value, index, var_type
                )
            }
        }
    }
}
