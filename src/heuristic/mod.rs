//! Heuristic solvers for HUBO problems.
//!
//! Provides lightweight metaheuristic solvers that work directly on the
//! polynomial objective without linearisation.  These are useful when an
//! exact solver is too slow or unavailable.
//!
//! # Available heuristics
//!
//! | Backend | Function | Description |
//! |---------|----------|-------------|
//! | SA      | [`sa::solve`] | Simulated Annealing with Boltzmann acceptance |
//! | Tabu    | [`tabu::solve`] | Tabu Search with short-term memory |
//! | Greedy  | [`greedy::solve`] | Steepest-descent local search with restarts |
//! | SAW     | [`saw::solve`] | Self-Avoiding Walk random exploration |
//! | Pool    | [`pool::solve`] | Diverse solution-pool hybrid heuristic |

pub mod greedy;
pub mod parallel_tempering;
pub mod pool;
pub mod sa;
pub mod saw;
pub mod tabu;

use std::fmt;
use std::io::{self, Write};
use std::path::Path;

use crate::coeff::Coeff;
use crate::domain::VarType;
pub use crate::solution::BitSolution;

// ---------------------------------------------------------------------------
// Shared result types
// ---------------------------------------------------------------------------

/// Termination status of a heuristic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The solver completed all iterations / cooling schedule.
    Completed,
    /// The solver stopped because the time limit was reached.
    TimeLimit,
    /// The solver stopped because a solution with objective ≤ cutoff was found.
    Cutoff,
    /// The solver stopped because SIGINT (Ctrl+C) was received.
    Interrupted,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Status::Completed => write!(f, "completed"),
            Status::TimeLimit => write!(f, "time limit"),
            Status::Cutoff => write!(f, "cutoff reached"),
            Status::Interrupted => write!(f, "interrupted"),
        }
    }
}

/// Result of a heuristic solve.
pub struct HeuristicResult<C: Coeff> {
    /// Name of the heuristic that produced this result (e.g. "SA", "Tabu").
    pub method: &'static str,
    /// The solver status.
    pub status: Status,
    /// Objective value of the best solution found.
    pub objective: C,
    /// Best solution found, stored as a bit-packed assignment.
    pub solution: BitSolution,
    /// Solving time in seconds.
    pub solving_time: f64,
    /// Time to solution — wall-clock seconds from start until the best
    /// incumbent was found.
    pub tts: f64,
    /// Total number of single-variable flips evaluated.
    pub iterations: u64,
}

impl<C: Coeff> HeuristicResult<C> {
    /// Write the solution to a file in HUES solution format.
    pub fn write_solution_file(&self, path: impl AsRef<Path>, var_type: VarType) -> io::Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let obj_f = self.objective.to_f64();
            let obj_val: serde_json::Value = if obj_f.is_finite() && obj_f.fract() == 0.0 {
                serde_json::Value::Number(serde_json::Number::from(obj_f as i64))
            } else {
                serde_json::Number::from_f64(obj_f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            };
            let obj = serde_json::json!({
                "status": self.status.to_string(),
                "method": self.method,
                "objective": obj_val,
                "time_s": self.solving_time,
                "tts_s": self.tts,
                "iterations": self.iterations,
                "solution": self.solution.to_json_array(var_type),
            });
            return std::fs::write(path, serde_json::to_string_pretty(&obj).unwrap());
        }

        let mut f = std::fs::File::create(path)?;

        writeln!(f, "# HUES solution file")?;
        writeln!(f, "STATUS {}", self.status)?;
        writeln!(f, "METHOD {}", self.method)?;
        writeln!(f, "OBJECTIVE {}", self.objective)?;
        writeln!(f, "BEST_BOUND n/a")?;
        writeln!(f, "TIME {:.6}", self.solving_time)?;
        writeln!(f, "TTS {:.6}", self.tts)?;
        writeln!(f, "ITERATIONS {}", self.iterations)?;

        writeln!(f, "SOLUTION")?;
        self.solution.write_to(&mut f, var_type)?;

        Ok(())
    }

    /// Print a formatted result block to stderr.
    pub fn print_report(&self, var_type: VarType) {
        let sol_str = self.solution.format_string(var_type);

        eprintln!();
        eprintln!("══════════════════════════════════════");
        eprintln!("  HUES Result  ({})", self.method);
        eprintln!("──────────────────────────────────────");
        eprintln!("  status     : {}", self.status);
        eprintln!("  objective  : {}", self.objective);
        eprintln!("  time       : {:.3} s", self.solving_time);
        eprintln!("  tts        : {:.3} s", self.tts);
        eprintln!("  iterations : {}", self.iterations);
        if !sol_str.is_empty() {
            eprintln!("  solution   : {sol_str}");
        }
        eprintln!("══════════════════════════════════════");
    }
}

// ---------------------------------------------------------------------------
// Shared configuration
// ---------------------------------------------------------------------------

/// Parameters common to all heuristic solvers.
#[derive(Default)]
pub struct CommonConfig {
    /// Time limit in seconds.  `None` means no limit.
    pub time_limit: Option<f64>,
    /// Cutoff value — stop as soon as objective ≤ cutoff.
    pub cutoff: Option<f64>,
    /// Optional seed for the PRNG.  `None` uses a time-based seed.
    pub seed: Option<u64>,
    /// If set, write the solution to this path after solving.
    pub solution_file: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers shared across heuristics
// ---------------------------------------------------------------------------

/// Format a solution as a compact string.
///
/// Delegates to [`BitSolution::format_string`]; kept for convenience.
pub fn format_solution(solution: &BitSolution, var_type: VarType) -> String {
    solution.format_string(var_type)
}

/// A fast, high-quality PRNG (xoshiro256**).
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to initialise state from a single seed.
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform random index in `0..n`.
    #[inline]
    pub fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0; // Safety: n should never be 0, but if it is, return 0
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform random f64 in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Generate a random feasible solution as a `BitSolution`.
///
/// Each bit is set (high value: BIN=1, SPIN=+1) or cleared (low value:
/// BIN=0, SPIN=−1) with equal probability.
pub(crate) fn random_solution(n: usize, _var_type: VarType, rng: &mut Rng) -> BitSolution {
    let mut bs = BitSolution::new(n);
    for i in 0..n {
        if rng.next_u64() & 1 == 1 {
            bs.values.insert(i);
        }
    }
    bs
}

/// Derive a base seed for the PRNG (from config or wall-clock).
pub(crate) fn base_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    })
}
