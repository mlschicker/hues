//! Incremental objective-evaluation state.

use crate::{coeff::Coeff, domain::VarDomain};

/// Cached single-variable objective deltas used by local-search heuristics.
///
/// `deltas[var]` equals the objective change if variable `var` were flipped
/// in the current solution represented by the paired [`TermState`].
pub struct DeltaCache<C: Coeff> {
    pub deltas: Vec<C>,
    pub marks: Vec<u32>,
    pub mark_epoch: u32,
}

/// Per-term status cache used for fast single-variable flip deltas.
///
/// `term_status[ti]` stores:
/// - **BIN**: `false` for 0 or `true` for 1 (whether the term monomial is active)
/// - **SPIN**: `false` for -1 or `true` for +1 (current sign of the term product)
///
/// `data` is auxiliary domain-specific state - for BIN it is the count of
/// "high" bits in the term; for SPIN it is `()`.
#[derive(Clone, Debug, PartialEq)]
pub struct TermState<V: VarDomain> {
    pub term_status: Vec<bool>,
    pub data: V::TermStateData,
}
