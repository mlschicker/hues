//! Exact solvers for HUBO problems.
//!
//! Currently provides a SCIP-based solver that linearises the polynomial
//! objective into a mixed-integer program.

pub mod bnb;
pub mod scip;

// Re-export the main types so callers can use `solver::SolveResult` etc.
pub use scip::{SolveResult, SolverConfig, solve, solve_mccormick, solve_nonlinear_constraint};
