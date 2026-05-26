//! Polynomial term representation.

use crate::coeff::Coeff;

/// A single weighted monomial term  c * v_{i1} * v_{i2} * ... * v_{ik}.
#[derive(Debug, Clone, PartialEq)]
pub struct Term<C: Coeff> {
    /// Variable indices (sorted, reduced: BIN=deduplicated, SPIN=pairwise reduced).
    pub indices: Vec<usize>,
    /// Coefficient.
    pub coeff: C,
}
