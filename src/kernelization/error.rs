use std::{error::Error, fmt};

use crate::{VarType, fixes::FixError};

#[derive(Debug, Clone)]
pub enum KernelizationError {
    InvalidVariableIndex {
        index: usize,
        n_vars: usize,
    },
    InvalidFixValue {
        index: usize,
        value: String,
        var_type: VarType,
    },
    ConflictingFixes {
        index: usize,
    },
    MappingMismatch {
        expected: usize,
        got: usize,
    },
    InvalidSolutionLength {
        expected: usize,
        got: usize,
    },
    LiftingError {
        source_fixed: usize,
        reduced_fixed: usize,
        expected_solution_length: usize,
    },
    FixingError {
        source: Box<FixError>,
    },
}

impl fmt::Display for KernelizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVariableIndex { index, n_vars } => {
                write!(f, "variable index {index} is out of range 0..{}", n_vars)
            }
            Self::InvalidFixValue {
                index,
                value,
                var_type,
            } => {
                write!(
                    f,
                    "invalid fixed value {value} for variable {index} with type {:?}",
                    var_type
                )
            }
            Self::ConflictingFixes { index } => {
                write!(f, "conflicting fixed values for variable {index}")
            }
            Self::MappingMismatch { expected, got } => {
                write!(
                    f,
                    "mapping mismatch: expected source size {expected}, got {got}"
                )
            }
            Self::InvalidSolutionLength { expected, got } => {
                write!(f, "invalid solution length: expected {expected}, got {got}")
            }
            Self::LiftingError {
                source_fixed,
                reduced_fixed,
                expected_solution_length,
            } => {
                write!(
                    f,
                    "lifting error: source has {source_fixed} fixed variables, reduced solution has {reduced_fixed} fixed variables, expected solution length {expected_solution_length}"
                )
            }
            Self::FixingError { source } => {
                write!(f, "error applying fix: {source}")
            }
        }
    }
}

impl Error for KernelizationError {}
