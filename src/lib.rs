pub mod bounds;
pub mod chordal_sdp;
pub mod coeff;
pub mod domain;
pub mod fixes;
pub mod heuristic;
pub mod instance;
pub mod interrupt;
pub mod kernelization;
pub mod lasserre;
pub mod model;
pub mod parser;
pub mod solution;
pub mod solver;
pub mod state;
pub mod term;
pub mod util;

// Re-export utilities at crate root for convenience.
pub use coeff::Coeff;
pub use domain::{Bin, Spin, VarDomain, VarType};
pub use instance::{HuboInstance, HuboInstanceEnum};
pub use solution::BitSolution;
pub use term::Term;
pub use util::error;

pub type Logger = ();
