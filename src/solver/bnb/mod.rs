//! Custom branch-and-bound solver for HUBO problems.

use std::collections::BinaryHeap;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json;

use crate::coeff::Coeff;
use crate::heuristic;
use crate::kernelization::{
    KernelizationConfig, KernelizationReport, Kernelizer, binary_roof_duality, spin_roof_duality,
};
use crate::solution::BitSolution;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
    term::Term,
};

mod branching;
mod constraints;
pub mod cutting_planes;
mod enumerate;
mod parallel;
mod probing;
mod reporting;
mod serial;
mod solve;
pub(crate) mod types;
mod util;

use crate::bounds::*;
use branching::*;
pub use constraints::*;
use cutting_planes::*;
use parallel::*;
use probing::*;
use reporting::*;
use serial::*;
use solve::*;
use types::*;

pub use crate::bounds::{
    Cheap, ChordalSdp, ClusterSubgradient, ExactLasserre, HittingSet, Lasserre, LowerBound,
    LpBound, Subgradient, Trwbp,
};
pub use solve::solve;
pub(crate) use types::ConstraintHandler;
pub use types::{Config, ProbingConfig, StrongBranchingConfig};
pub use types::{Node, PartiallyAssignedTerm};
