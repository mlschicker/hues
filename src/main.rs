#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
use std::sync::Arc;

use std::fs;
use std::io;
use std::process;
use std::sync::mpsc;

use chrono::Local;
use clap::{Parser, ValueEnum};
use env_logger::{Builder, Target};

use hues::Coeff;
use hues::Logger;
use hues::bounds::{Cheap, ChordalSdp, ClusterSubgradient, ExactLasserre, HittingSet, Lasserre, LpBound, RltLp, SheraliAdams, Subgradient, Trwbp};
use hues::bounds::lp::LpConfig;
use hues::bounds::rlt_lagrangian::RltConfig;
use hues::bounds::sherali_adams::SheraliAdamsConfig;
use hues::heuristic;
use hues::instance::{HuboInstance, HuboInstanceEnum};
use hues::kernelization::KernelizationConfig;
use hues::lasserre::{ExactLasserreConfig, LasserreConfig};
use hues::parser;
use hues::solver;
use hues::solver::bnb::cutting_planes::CoverCutDomain;


/// Which solver backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SolverBackend {
    /// Exact solver via SCIP (McCormick linearisation)
    ScipMc,
    /// Exact solver via SCIP (full nonlinear objective constraint via FFI)
    ScipNl,
    /// Exact custom branch-and-bound solver
    Bnb,
    /// Simulated annealing heuristic
    Sa,
    /// Tabu search heuristic
    Tabu,
    /// Greedy steepest-descent heuristic with restarts
    Greedy,
    /// Self-avoiding walk heuristic
    Saw,
    /// Diverse solution-pool hybrid heuristic
    Pool,
    /// Parallel tempering heuristic
    Pt,
}

/// Numeric type used for coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CoeffType {
    /// Auto-detect: try integer first, fall back to float
    Auto,
    /// Exact integer arithmetic (i64) — eliminates floating-point errors
    Int,
    /// Floating-point arithmetic (f64)
    Float,
}

/// CLI log verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Target objective domain/file format for conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ConvertTarget {
    /// Convert to HUBO (binary variables, HUBO 1 magic)
    Hubo,
    /// Convert to HUSO (spin variables, HUSO 1 magic)
    Huso,
    /// Emit the instance as structured JSON
    Json,
}

impl LogLevel {
    fn as_filter(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

// ── BnB config file ──────────────────────────────────────────────────────────

/// All fields are optional; unset fields fall back to their hardcoded defaults.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BnbFileConfig {
    lb_method: Option<String>,
    threads: Option<usize>,
    node_limit: Option<u64>,
    optimality_tol: Option<f64>,
    warm_start: Option<bool>,
    warm_start_time: Option<f64>,
    kernelization: Option<bool>,
    node_kernelization: Option<bool>,
    log_every_nodes: Option<u64>,
    subgrad_max_iter: Option<usize>,
    subgrad_step_size: Option<f64>,
    subgrad_step_decay: Option<f64>,
    trwbp_max_iter: Option<usize>,
    trwbp_damping: Option<f64>,
    hs_max_cores: Option<usize>,
    hs_max_search_nodes: Option<usize>,
    lasserre_order: Option<usize>,
    lasserre_max_vars: Option<usize>,
    exact_lasserre_order: Option<usize>,
    exact_lasserre_max_vars: Option<usize>,
    cluster_max_vars: Option<usize>,
}

struct BnbResolvedConfig {
    lb_method: String,
    threads: usize,
    node_limit: Option<u64>,
    optimality_tol: f64,
    warm_start: bool,
    warm_start_time: f64,
    kernelization: bool,
    node_kernelization: bool,
    log_every_nodes: u64,
    subgrad_max_iter: usize,
    subgrad_step_size: f64,
    subgrad_step_decay: f64,
    trwbp_max_iter: usize,
    trwbp_damping: f64,
    hs_max_cores: usize,
    hs_max_search_nodes: usize,
    lasserre_order: usize,
    lasserre_max_vars: usize,
    exact_lasserre_order: usize,
    exact_lasserre_max_vars: usize,
    cluster_max_vars: usize,
}

fn resolve_bnb_config(cli: &Cli, file: &BnbFileConfig) -> BnbResolvedConfig {
    BnbResolvedConfig {
        lb_method: cli
            .bnb_lb_method
            .clone()
            .or_else(|| file.lb_method.clone())
            .unwrap_or_else(|| "cheap".to_string()),
        threads: cli.bnb_threads.or(file.threads).unwrap_or(1),
        node_limit: cli.bnb_node_limit.or(file.node_limit),
        optimality_tol: cli
            .bnb_optimality_tol
            .or(file.optimality_tol)
            .unwrap_or(1e-5),
        warm_start: if cli.bnb_no_heuristic_warmstart {
            false
        } else {
            file.warm_start.unwrap_or(true)
        },
        warm_start_time: cli
            .bnb_heuristic_time
            .or(file.warm_start_time)
            .unwrap_or(0.5),
        kernelization: if cli.bnb_no_kernelization {
            false
        } else {
            file.kernelization.unwrap_or(true)
        },
        node_kernelization: if cli.bnb_no_node_kernelization {
            false
        } else {
            file.node_kernelization.unwrap_or(true)
        },
        log_every_nodes: cli
            .bnb_log_every_nodes
            .or(file.log_every_nodes)
            .unwrap_or(200),
        subgrad_max_iter: cli
            .bnb_subgrad_max_iter
            .or(file.subgrad_max_iter)
            .unwrap_or(64),
        subgrad_step_size: cli
            .bnb_subgrad_step_size
            .or(file.subgrad_step_size)
            .unwrap_or(1.0),
        subgrad_step_decay: cli
            .bnb_subgrad_step_decay
            .or(file.subgrad_step_decay)
            .unwrap_or(1.0),
        trwbp_max_iter: cli.bnb_trwbp_max_iter.or(file.trwbp_max_iter).unwrap_or(8),
        trwbp_damping: cli.bnb_trwbp_damping.or(file.trwbp_damping).unwrap_or(0.5),
        hs_max_cores: cli.bnb_hs_max_cores.or(file.hs_max_cores).unwrap_or(64),
        hs_max_search_nodes: cli
            .bnb_hs_max_search_nodes
            .or(file.hs_max_search_nodes)
            .unwrap_or(50_000),
        lasserre_order: cli.bnb_lasserre_order.or(file.lasserre_order).unwrap_or(1),
        lasserre_max_vars: cli
            .bnb_lasserre_max_vars
            .or(file.lasserre_max_vars)
            .unwrap_or(30),
        exact_lasserre_order: cli
            .bnb_exact_lasserre_order
            .or(file.exact_lasserre_order)
            .unwrap_or(1),
        exact_lasserre_max_vars: cli
            .bnb_exact_lasserre_max_vars
            .or(file.exact_lasserre_max_vars)
            .unwrap_or(20),
        cluster_max_vars: cli
            .bnb_cluster_max_vars
            .or(file.cluster_max_vars)
            .unwrap_or(15),
    }
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            fs::read_to_string("/etc/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("model name\t: ")
                    .or_else(|| line.strip_prefix("Hardware\t: "))
                    .or_else(|| line.strip_prefix("Processor\t: "))
                    .map(|value| value.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn log_startup_banner() {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S %Z");
    let logical_cpus = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    ::log::info!("date/time : {}", timestamp);
    ::log::info!(
        "hardware  : host={}, os={}, arch={}, logical_cpus={}, cpu_model={}",
        hostname(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        logical_cpus,
        cpu_model()
    );
    ::log::info!("solver    : hues {}", env!("CARGO_PKG_VERSION"));
    ::log::info!(
        "command   : {}",
        std::env::args().collect::<Vec<_>>().join(" ")
    );
    ::log::info!("");
}

/// Writes log records to both stderr and a file simultaneously.
struct DualWriter {
    file: fs::File,
}

impl io::Write for DualWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        io::stderr().write_all(buf).ok();
        self.file.write_all(buf).ok();
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush().ok();
        self.file.flush()
    }
}

fn init_logger(level: LogLevel, log_file: Option<&str>) -> io::Result<()> {
    let mut builder = Builder::new();
    builder.parse_filters(level.as_filter());
    // Records tagged target: "table" are printed as-is (no level prefix) so that
    // box-drawing table rows and the result banner stay visually aligned.
    builder.format(|buf, record| {
        use std::io::Write;
        if record.target() == "table" {
            writeln!(buf, "{}", record.args())
        } else {
            writeln!(buf, "[{}] {}", record.level(), record.args())
        }
    });

    if let Some(path) = log_file {
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        // DualWriter tees every log record to both stderr and the file so the
        // terminal always shows the table and the log file is complete.
        builder.target(Target::Pipe(Box::new(DualWriter { file })));
    }

    let _ = builder.try_init();
    Ok(())
}

/// HUES — HUBO solver
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Path to the HUBO-TL input file
    file: String,

    /// Run kernelization only (no solving) on `file`.
    /// If `file` is a directory, all supported instance files are scanned recursively.
    #[arg(long)]
    kernelization_scan: bool,

    /// Output CSV path for --kernelization-scan mode.
    #[arg(long, requires = "kernelization_scan")]
    kernelization_scan_csv: Option<String>,

    /// Solver backend to use
    #[arg(long, default_value = "bnb")]
    solver: SolverBackend,

    // ---- General solver options ---------------------------------------------
    /// Coefficient type: auto, int, or float
    #[arg(long, default_value = "auto")]
    coeff_type: CoeffType,

    /// Time limit in seconds
    #[arg(short = 't', long)]
    time_limit: Option<f64>,

    /// Stop when a solution with objective ≤ this value is found
    #[arg(short = 'c', long)]
    cutoff: Option<f64>,

    /// Write the solution to this file after solving
    #[arg(short = 's', long)]
    solution_file: Option<String>,

    /// Provide a solution file as a warm-start hint
    #[arg(short = 'i', long)]
    initial_solution: Option<String>,

    // ---- Logging options ----------------------------------------------------
    /// HUES log verbosity: trace, debug, info, warn, error
    #[arg(short = 'v', long, default_value = "info")]
    log_level: LogLevel,

    /// Write a detailed log to this file
    #[arg(short = 'l', long)]
    log_file: Option<String>,

    /// Only parse and print the instance; do not solve
    #[arg(long)]
    parse_only: bool,

    /// Convert input objective and write the converted instance to --output
    #[arg(long, value_enum)]
    convert_to: Option<ConvertTarget>,

    /// Output file path for conversion mode
    #[arg(short = 'o', long, requires = "convert_to")]
    output: Option<String>,

    // ---- SCIP-specific options ----------------------------------------------
    /// Maximum number of branch-and-bound nodes (SCIP only)
    #[arg(long)]
    scip_node_limit: Option<i64>,

    /// Relative MIP gap limit, e.g. 0.01 for 1% (SCIP only)
    #[arg(long)]
    scip_gap: Option<f64>,

    /// Number of threads for concurrent solving (SCIP only)
    #[arg(long)]
    scip_threads: Option<i32>,

    /// SCIP verbosity level (0 = silent, 4 = default, 5 = full)
    #[arg(long, default_value_t = 4)]
    scip_verbosity: i32,

    // ---- BnB-specific options ---------------------------------------------
    /// Path to a TOML config file for advanced BnB parameters
    #[arg(long)]
    config: Option<String>,

    /// Maximum number of branch-and-bound nodes (BnB only)
    #[arg(long)]
    bnb_node_limit: Option<u64>,

    /// RNG seed for warm-start heuristics (default 0 for determinism)
    #[arg(long, default_value_t = 0)]
    bnb_seed: u64,

    /// Minimum bound improvement to log, as a percentage of the gap (default 1.0)
    #[arg(long, default_value_t = 1.0)]
    bnb_bound_log_pct: f64,

    /// BnB progress log interval in explored nodes (0 disables; default 5000)
    #[arg(long)]
    bnb_log_every_nodes: Option<u64>,

    /// Append one BnB run summary row to this CSV file (benchmark mode)
    #[arg(long)]
    bnb_stats_csv: Option<String>,

    /// Number of parallel worker threads for BnB (default 1 = serial)
    #[arg(long)]
    bnb_threads: Option<usize>,

    /// Disable heuristic warm-starts before BnB search
    #[arg(long)]
    bnb_no_heuristic_warmstart: bool,

    /// Disable root-level kernelization
    #[arg(long)]
    bnb_no_kernelization: bool,

    /// Disable node-level kernelization
    #[arg(long)]
    bnb_no_node_kernelization: bool,

    /// Absolute optimality tolerance: prune a node when lb + tol >= ub (default 1e-5)
    #[arg(long)]
    bnb_optimality_tol: Option<f64>,

    /// Per-heuristic time budget in seconds for BnB warm-start heuristics (default 0.5)
    #[arg(long)]
    bnb_heuristic_time: Option<f64>,

    // ---- BnB lower-bound options ------------------------------------------
    /// Lower-bounding oracle for BnB: cheap, trwbp, hittingset, subgradient,
    /// lasserre, chordal-sdp, exact-lasserre (default: cheap)
    #[arg(long)]
    bnb_lb_method: Option<String>,

    /// Maximum subgradient ascent iterations (BnB+subgradient only; default 64)
    #[arg(long)]
    bnb_subgrad_max_iter: Option<usize>,

    /// Polyak relaxation factor beta in (0,2] (BnB+subgradient only; default 1.0)
    #[arg(long)]
    bnb_subgrad_step_size: Option<f64>,

    /// Per-iteration decay applied to beta in (0,1] (BnB+subgradient only; default 1.0)
    #[arg(long)]
    bnb_subgrad_step_decay: Option<f64>,

    /// Number of TRWBP message-passing sweeps (BnB+trwbp only; default 8)
    #[arg(long)]
    bnb_trwbp_max_iter: Option<usize>,

    /// Damping factor for TRWBP updates in [0,1) (BnB+trwbp only; default 0.5)
    #[arg(long)]
    bnb_trwbp_damping: Option<f64>,

    /// Maximum number of unsat cores in hitting-set LB (BnB+hittingset only; default 64)
    #[arg(long)]
    bnb_hs_max_cores: Option<usize>,

    /// Maximum search nodes for internal hitting-set solver (BnB+hittingset only; default 50000)
    #[arg(long)]
    bnb_hs_max_search_nodes: Option<usize>,

    /// Lasserre hierarchy order d (BnB+lasserre only; default 1)
    #[arg(long)]
    bnb_lasserre_order: Option<usize>,

    /// Legacy option kept for compatibility (ignored by SCIP-backed Lasserre)
    #[arg(long)]
    bnb_lasserre_max_iter: Option<usize>,

    /// Only apply Lasserre SDP when unassigned variables ≤ this limit (BnB+lasserre only; default 30)
    #[arg(long)]
    bnb_lasserre_max_vars: Option<usize>,

    /// Legacy option kept for compatibility (ignored by SCIP-backed Lasserre)
    #[arg(long)]
    bnb_lasserre_step_size: Option<f64>,

    /// Hierarchy order d for exact Lasserre SDP (BnB+exact-lasserre only; default 1)
    #[arg(long)]
    bnb_exact_lasserre_order: Option<usize>,

    /// Max free variables for exact Lasserre SDP (BnB+exact-lasserre only; default 20)
    #[arg(long)]
    bnb_exact_lasserre_max_vars: Option<usize>,

    /// Max distinct free variables per cluster for cluster-subgradient LB (default 15, hard cap 20)
    #[arg(long)]
    bnb_cluster_max_vars: Option<usize>,

    // ---- SA-specific options ------------------------------------------------
    /// Initial temperature (SA only)
    #[arg(long, default_value_t = 10.0)]
    sa_temp: f64,

    /// Final temperature (SA only)
    #[arg(long, default_value_t = 1e-6)]
    sa_final_temp: f64,

    /// Cooling rate (SA only, typical 0.99–0.9999)
    #[arg(long, default_value_t = 0.9995)]
    sa_cooling: f64,

    /// Number of independent restarts (SA only)
    #[arg(long, default_value_t = 10)]
    sa_restarts: usize,

    /// PRNG seed (SA only, omit for time-based seed)
    #[arg(long)]
    sa_seed: Option<u64>,

    // ---- Tabu-specific options ----------------------------------------------
    /// Tabu tenure — iterations a variable stays tabu (Tabu only, default √n)
    #[arg(long)]
    tabu_tenure: Option<usize>,

    /// Maximum iterations per restart (Tabu only, default: unbounded)
    #[arg(long)]
    tabu_max_iter: Option<u64>,

    /// Number of independent restarts (Tabu only)
    #[arg(long, default_value_t = 1)]
    tabu_restarts: usize,

    /// PRNG seed (Tabu only, omit for time-based seed)
    #[arg(long)]
    tabu_seed: Option<u64>,

    // ---- Greedy-specific options --------------------------------------------
    /// Number of independent restarts (Greedy only)
    #[arg(long, default_value_t = 1)]
    greedy_restarts: usize,

    /// Maximum improving flips per restart (Greedy only, default: unbounded)
    #[arg(long)]
    greedy_max_flips: Option<u64>,

    /// PRNG seed (Greedy only, omit for time-based seed)
    #[arg(long)]
    greedy_seed: Option<u64>,

    // ---- SAW-specific options -----------------------------------------------
    /// Number of independent walks (SAW only)
    #[arg(long, default_value_t = 1)]
    saw_n_walks: usize,

    /// Maximum steps per walk (SAW only, default: unbounded)
    #[arg(long)]
    saw_max_steps: Option<u64>,

    /// Apply steepest-descent polish at end of each segment (SAW only)
    #[arg(long)]
    saw_local_search: bool,

    /// PRNG seed (SAW only, omit for time-based seed)
    #[arg(long)]
    saw_seed: Option<u64>,

    // ---- Pool-specific options ----------------------------------------------
    /// Maximum number of solutions in the pool (Pool only)
    #[arg(long, default_value_t = 16)]
    pool_size: usize,

    /// Number of initial random solutions sampled (Pool only)
    #[arg(long, default_value_t = 32)]
    pool_init: usize,

    /// Maximum offspring iterations (Pool only, default: unbounded)
    #[arg(long)]
    pool_max_iter: Option<u64>,

    /// Maximum number of differing bits flipped by XOR move (Pool only)
    #[arg(long, default_value_t = 8)]
    pool_xor_max_flips: usize,

    /// PRNG seed (Pool only, omit for time-based seed)
    #[arg(long)]
    pool_seed: Option<u64>,

    // ---- Parallel Tempering-specific options ------------------------------
    /// Replicas per run (PT only)
    #[arg(long, default_value_t = 12)]
    pt_replicas: usize,

    /// Number of independent parallel runs (PT only)
    #[arg(long, default_value_t = 8)]
    pt_runs: usize,

    /// Number of sweeps per run (PT only)
    #[arg(long, default_value_t = 10_000)]
    pt_sweeps: usize,

    /// Attempt swaps every this many sweeps (PT only)
    #[arg(long, default_value_t = 5)]
    pt_swap_interval: usize,

    /// Minimum temperature (cold replica, PT only)
    #[arg(long, default_value_t = 0.1)]
    pt_t_min: f64,

    /// Maximum temperature (hot replica, PT only)
    #[arg(long, default_value_t = 10.0)]
    pt_t_max: f64,

    /// Disable greedy descent after accepted swaps (PT only)
    #[arg(long)]
    pt_no_greedy_after_swap: bool,

    /// Adapt ladder every this many sweeps; 0 disables adaptation (PT only)
    #[arg(long, default_value_t = 500)]
    pt_adapt_interval: usize,

    /// Target swap acceptance rate for adaptation (PT only)
    #[arg(long, default_value_t = 0.25)]
    pt_target_accept_rate: f64,

    /// PRNG seed (PT only, omit for time-based seed)
    #[arg(long)]
    pt_seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// Generic run function — monomorphised for i64 and f64
// ---------------------------------------------------------------------------

fn run_dispatch<C: Coeff>(instance: HuboInstanceEnum<C>, cli: &Cli, logger: &Logger) {
    match instance {
        HuboInstanceEnum::Bin(i) => run(i, cli, logger),
        HuboInstanceEnum::Spin(i) => run(i, cli, logger),
    }
}

fn run<C: Coeff, V: CoverCutDomain>(instance: HuboInstance<C, V>, cli: &Cli, logger: &Logger) {
    let instance = Arc::new(instance);

    ::log::info!(
        "instance: var_type={:?}  n_vars={}  n_terms={}  offset={}  coeff_type={}",
        instance.var_type(),
        instance.n_vars(),
        instance.n_terms(),
        instance.offset,
        C::type_name()
    );

    ::log::info!("");

    if cli.parse_only {
        ::log::info!("parse-only mode");
        ::log::info!("  var_type    = {:?}", instance.var_type());
        ::log::info!("  n_vars      = {}", instance.n_vars());
        ::log::info!("  n_terms     = {}", instance.n_terms());
        ::log::info!("  offset      = {}", instance.offset);
        ::log::info!("  coeff_type  = {}", C::type_name());
        for (i, term) in instance.terms.iter().enumerate() {
            ::log::info!("  [{i}] indices={:?}  coeff={}", term.indices, term.coeff);
        }
        return;
    }

    // ---- Load initial solution if provided ---------------------------------
    let init_solution = if let Some(ref sol_path) = cli.initial_solution {
        log::info!("loading initial solution from {}", sol_path);
        match parser::parse_solution_file::<C>(sol_path, instance.n_vars(), instance.var_type()) {
            Ok(vals) => {
                log::info!(
                    "initial solution loaded with value {}",
                    instance.evaluate(&vals)
                );
                Some(vals)
            }
            Err(e) => {
                log::warn!("failed to load initial solution: {}", e);
                None
            }
        }
    } else {
        None
    };

    // ---- Solve -------------------------------------------------------------
    match cli.solver {
        SolverBackend::ScipMc => {
            let config = solver::SolverConfig {
                time_limit: cli.time_limit.map(|t| t as usize),
                node_limit: cli.scip_node_limit,
                gap_limit: cli.scip_gap,
                verbosity: cli.scip_verbosity,
                threads: cli.scip_threads,
                solution_file: cli.solution_file.clone(),
            };

            let result = solver::solve_mccormick(&instance, &config, logger, init_solution);

            let status_str = format!("{:?}", result.status);
            let obj_str = match result.objective {
                Some(obj) => format!("{obj}"),
                None => "n/a".to_string(),
            };
            let sol_str = match &result.solution {
                Some(sol) => sol.format_string(instance.var_type()),
                None => String::new(),
            };

            eprintln!();
            eprintln!("══════════════════════════════════════");
            eprintln!("  HUES Result  (SCIP MC)");
            eprintln!("──────────────────────────────────────");
            eprintln!("  status     : {status_str}");
            eprintln!("  objective  : {obj_str}");
            eprintln!("  best bound : {}", result.best_bound);
            eprintln!("  time       : {:.3} s", result.solving_time);
            let tts_str = match result.tts {
                Some(t) => format!("{:.3} s", t),
                None => "n/a".to_string(),
            };
            eprintln!("  tts        : {tts_str}");
            eprintln!("  nodes      : {}", result.n_nodes);
            if !sol_str.is_empty() {
                eprintln!("  solution   : {sol_str}");
            }
            eprintln!("══════════════════════════════════════");
        }
        SolverBackend::ScipNl => {
            let config = solver::SolverConfig {
                time_limit: cli.time_limit.map(|t| t as usize),
                node_limit: cli.scip_node_limit,
                gap_limit: cli.scip_gap,
                verbosity: cli.scip_verbosity,
                threads: cli.scip_threads,
                solution_file: cli.solution_file.clone(),
            };

            let result =
                solver::solve_nonlinear_constraint(&instance, &config, logger, init_solution);

            let status_str = format!("{:?}", result.status);
            let obj_str = match result.objective {
                Some(obj) => format!("{obj}"),
                None => "n/a".to_string(),
            };
            let sol_str = match &result.solution {
                Some(sol) => sol.format_string(instance.var_type()),
                None => String::new(),
            };

            eprintln!();
            eprintln!("══════════════════════════════════════");
            eprintln!("  HUES Result  (SCIP NL)");
            eprintln!("──────────────────────────────────────");
            eprintln!("  status     : {status_str}");
            eprintln!("  objective  : {obj_str}");
            eprintln!("  best bound : {}", result.best_bound);
            eprintln!("  time       : {:.3} s", result.solving_time);
            let tts_str = match result.tts {
                Some(t) => format!("{:.3} s", t),
                None => "n/a".to_string(),
            };
            eprintln!("  tts        : {tts_str}");
            eprintln!("  nodes      : {}", result.n_nodes);
            if !sol_str.is_empty() {
                eprintln!("  solution   : {sol_str}");
            }
            eprintln!("══════════════════════════════════════");
        }
        SolverBackend::Bnb => {
            let file_cfg: BnbFileConfig = cli
                .config
                .as_ref()
                .map(|path| {
                    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
                        log::error!("cannot read config file {path}: {e}");
                        process::exit(1);
                    });
                    toml::from_str::<BnbFileConfig>(&content).unwrap_or_else(|e| {
                        log::error!("invalid config file {path}: {e}");
                        process::exit(1);
                    })
                })
                .unwrap_or_default();

            let bnb_cfg = resolve_bnb_config(cli, &file_cfg);

            macro_rules! run_bnb {
                ($lb:expr) => {{
                    let config = solver::bnb::Config {
                        lb: $lb,
                        time_limit: cli.time_limit,
                        node_limit: bnb_cfg.node_limit,
                        cutoff: cli.cutoff,
                        progress_every_nodes: (bnb_cfg.log_every_nodes > 0)
                            .then_some(bnb_cfg.log_every_nodes),
                        stats_csv: cli.bnb_stats_csv.clone(),
                        instance_name: Some(cli.file.clone()),
                        solution_file: cli.solution_file.clone(),
                        warm_start_heuristics: bnb_cfg.warm_start,
                        warm_start_heuristic_time_limit: (bnb_cfg.warm_start_time > 0.0)
                            .then_some(bnb_cfg.warm_start_time),
                        n_threads: bnb_cfg.threads,
                        node_kernelization: if bnb_cfg.node_kernelization {
                            KernelizationConfig::node_level()
                        } else {
                            KernelizationConfig::none()
                        },
                        probing: solver::bnb::ProbingConfig::default(),
                        strong_branching: solver::bnb::StrongBranchingConfig::default(),
                        optimality_tol: bnb_cfg.optimality_tol,
                        kernelization: bnb_cfg.kernelization,
                        seed: cli.bnb_seed,
                        bound_log_min_improvement_pct: cli.bnb_bound_log_pct,
                    };
                    let (solution_sender, solution_receiver) = mpsc::channel();
                    let result =
                        solver::bnb::solve(&instance, &config, init_solution, &solution_receiver);
                    drop(solution_sender);
                    drop(solution_receiver);
                    result
                }};
            }

            let result = match bnb_cfg.lb_method.to_lowercase().as_str() {
                "lasserre" | "sdp" => run_bnb!(Lasserre(LasserreConfig {
                    level: bnb_cfg.lasserre_order,
                    max_iter: cli.bnb_lasserre_max_iter.unwrap_or(100),
                    max_vars: bnb_cfg.lasserre_max_vars,
                    max_basis: 80,
                    step_size: cli.bnb_lasserre_step_size.unwrap_or(0.1),
                })),
                "chordal-sdp" | "chordal_sdp" | "chordal" => {
                    run_bnb!(ChordalSdp(LasserreConfig {
                        level: bnb_cfg.lasserre_order,
                        max_iter: cli.bnb_lasserre_max_iter.unwrap_or(100),
                        max_vars: bnb_cfg.lasserre_max_vars,
                        max_basis: 80,
                        step_size: cli.bnb_lasserre_step_size.unwrap_or(0.1),
                    }))
                }
                "trwbp" | "trw-bp" | "trw_bp" => run_bnb!(Trwbp {
                    max_iter: bnb_cfg.trwbp_max_iter,
                    damping: bnb_cfg.trwbp_damping.clamp(0.0, 0.999),
                }),
                "hittingset" | "hitting-set" | "hitting_set" | "hs" => run_bnb!(HittingSet {
                    max_cores: bnb_cfg.hs_max_cores.max(1),
                    max_search_nodes: bnb_cfg.hs_max_search_nodes.max(1),
                }),
                "subgradient" | "subgrad" | "sg" => run_bnb!(Subgradient {
                    max_iter: bnb_cfg.subgrad_max_iter.max(1),
                    step_size: bnb_cfg.subgrad_step_size.max(0.0),
                    step_decay: bnb_cfg.subgrad_step_decay.clamp(0.0, 1.0),
                    optimality_tol: bnb_cfg.optimality_tol.max(0.0),
                }),
                "cluster-subgradient" | "cluster_subgradient" | "cluster-sg" | "csg" => {
                    run_bnb!(ClusterSubgradient {
                        max_iter: bnb_cfg.subgrad_max_iter.max(1),
                        step_size: bnb_cfg.subgrad_step_size.max(0.0),
                        step_decay: bnb_cfg.subgrad_step_decay.clamp(0.0, 1.0),
                        optimality_tol: bnb_cfg.optimality_tol.max(0.0),
                        max_cluster_vars: bnb_cfg.cluster_max_vars,
                    })
                }
                "exact-lasserre" | "exact_lasserre" | "exact-sdp" | "exact_sdp" => {
                    run_bnb!(ExactLasserre(ExactLasserreConfig {
                        level: bnb_cfg.exact_lasserre_order,
                        max_vars: bnb_cfg.exact_lasserre_max_vars,
                        max_basis: 80,
                    }))
                }
                "rlt" | "rlt-lp" | "rlt_lp" => run_bnb!(RltLp(RltConfig {
                    max_vars: bnb_cfg.lasserre_max_vars,
                })),
                "sherali-adams" | "sherali_adams" | "sa-lp" | "sa_lp" => {
                    run_bnb!(SheraliAdams(SheraliAdamsConfig {
                        max_vars: bnb_cfg.lasserre_max_vars,
                        ..SheraliAdamsConfig::default()
                    }))
                }
                "lp" | "lp-bound" | "lp_bound" | "mccormick-lp" => {
                    run_bnb!(LpBound(LpConfig {
                        max_cols: bnb_cfg.lasserre_max_vars,
                    }))
                }
                _ => run_bnb!(Cheap),
            };

            let status_str = format!("{:?}", result.status);
            let obj_str = match result.objective {
                Some(obj) => format!("{obj}"),
                None => "n/a".to_string(),
            };
            let sol_str = match &result.solution {
                Some(sol) => sol.format_string(instance.var_type()),
                None => String::new(),
            };

            let tts_str = match result.tts {
                Some(t) => format!("{:.3} s", t),
                None => "n/a".to_string(),
            };
            log::info!(target: "table", "");
            log::info!(target: "table", "══════════════════════════════════════");
            log::info!(target: "table", "  HUES Result  (BnB)");
            log::info!(target: "table", "──────────────────────────────────────");
            log::info!(target: "table", "  status     : {status_str}");
            log::info!(target: "table", "  objective  : {obj_str}");
            log::info!(target: "table", "  best bound : {}", result.best_bound);
            log::info!(target: "table", "  time       : {:.3} s", result.solving_time);
            log::info!(target: "table", "  tts        : {tts_str}");
            log::info!(target: "table", "  explored   : {}", result.n_nodes);
            log::info!(target: "table", "  unexplored : {}", result.unexplored_nodes);
            log::info!(target: "table", "  pruned     : {}", result.pruned_nodes);
            if !sol_str.is_empty() {
                log::info!(target: "table", "  solution   : {sol_str}");
            }
            log::info!(target: "table", "══════════════════════════════════════");
        }
        SolverBackend::Sa => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.sa_seed,
                solution_file: cli.solution_file.clone(),
            };
            let config = heuristic::sa::Config {
                common,
                initial_temp: cli.sa_temp,
                final_temp: cli.sa_final_temp,
                cooling_rate: cli.sa_cooling,
                restarts: cli.sa_restarts,
                ..Default::default()
            };

            let init = init_solution
                .as_ref()
                .map(|v| heuristic::BitSolution::from_vec(v));
            let result = heuristic::sa::solve(&instance, &config, init.as_ref());
            result.print_report(instance.var_type());
        }
        SolverBackend::Tabu => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.tabu_seed,
                solution_file: cli.solution_file.clone(),
            };
            let config = heuristic::tabu::Config {
                common,
                tenure: cli.tabu_tenure,
                max_iterations: cli.tabu_max_iter,
                restarts: cli.tabu_restarts,
            };

            let init = init_solution
                .as_ref()
                .map(|v| heuristic::BitSolution::from_vec(v));
            let result = heuristic::tabu::solve(&instance, &config, init.as_ref());
            result.print_report(instance.var_type());
        }
        SolverBackend::Greedy => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.greedy_seed,
                solution_file: cli.solution_file.clone(),
            };
            let config = heuristic::greedy::Config {
                common,
                restarts: cli.greedy_restarts,
                max_flips: cli.greedy_max_flips,
            };

            let result = heuristic::greedy::solve(&instance, &config, logger);
            result.print_report(instance.var_type());
        }
        SolverBackend::Saw => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.saw_seed,
                solution_file: cli.solution_file.clone(),
            };
            let config = heuristic::saw::Config {
                common,
                n_walks: cli.saw_n_walks,
                max_steps: cli.saw_max_steps,
                local_search: cli.saw_local_search,
            };

            let result = heuristic::saw::solve(&instance, &config, logger);
            result.print_report(instance.var_type());
        }
        SolverBackend::Pool => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.pool_seed,
                solution_file: cli.solution_file.clone(),
            };
            let config = heuristic::pool::Config {
                common,
                pool_size: cli.pool_size,
                init_solutions: cli.pool_init,
                max_iterations: cli.pool_max_iter,
                xor_max_flips: cli.pool_xor_max_flips,
            };

            let result = heuristic::pool::solve(&instance, &config, logger);
            result.print_report(instance.var_type());
        }
        SolverBackend::Pt => {
            let common = heuristic::CommonConfig {
                time_limit: cli.time_limit,
                cutoff: cli.cutoff,
                seed: cli.pt_seed,
                solution_file: cli.solution_file.clone(),
            };

            let config = heuristic::parallel_tempering::Config {
                common,
                n_replicas: cli.pt_replicas,
                n_runs: cli.pt_runs,
                n_sweeps: cli.pt_sweeps,
                swap_interval: cli.pt_swap_interval,
                t_min: cli.pt_t_min,
                t_max: cli.pt_t_max,
                greedy_after_swap: !cli.pt_no_greedy_after_swap,
                adapt_interval: cli.pt_adapt_interval,
                target_accept_rate: cli.pt_target_accept_rate,
            };

            let result = heuristic::parallel_tempering::solve(&instance, &config, init_solution);
            result.print_report(instance.var_type());
        }
    }
}

fn main() {

    let cli = Cli::parse();

    // ---- Set up logger -----------------------------------------------------
    init_logger(cli.log_level, cli.log_file.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error initializing logger: {e}");
        process::exit(1);
    });
    print_banner();
    log_startup_banner();
    let logger: Logger = ();

    // ---- Install Ctrl+C handler --------------------------------------------
    hues::interrupt::install_handler();


    // ---- Read & parse ------------------------------------------------------
    ::log::info!("reading {}", cli.file);

    let contents = fs::read_to_string(&cli.file).unwrap_or_else(|e| {
        ::log::error!("cannot read {}: {e}", cli.file);
        process::exit(1);
    });

    if let Some(target) = cli.convert_to {
        let output_path = cli.output.clone().unwrap_or_else(|| {
            ::log::error!("--output is required when using --convert-to");
            process::exit(1);
        });

        let (instance, metadata) = match parser::parse_auto::<f64>(&contents, Some(&cli.file)) {
            Ok(res) => res,
            Err(e) => {
                ::log::error!("parse error: {e}");
                process::exit(1);
            }
        };

        let text = match (target, instance) {
            (ConvertTarget::Hubo, HuboInstanceEnum::Bin(i)) => i.to_hubo().to_hubo_tl(None),
            (ConvertTarget::Hubo, HuboInstanceEnum::Spin(i)) => i.to_hubo().to_hubo_tl(None),
            (ConvertTarget::Huso, HuboInstanceEnum::Bin(i)) => i.to_huso().to_huso_tl(None),
            (ConvertTarget::Huso, HuboInstanceEnum::Spin(i)) => i.to_huso().to_huso_tl(None),
            (ConvertTarget::Json, HuboInstanceEnum::Bin(i)) => i.to_json(Some(metadata)),
            (ConvertTarget::Json, HuboInstanceEnum::Spin(i)) => i.to_json(Some(metadata)),
        };

        if let Err(e) = fs::write(&output_path, text) {
            ::log::error!("cannot write {}: {e}", output_path);
            process::exit(1);
        }

        ::log::info!("wrote converted instance to {}", output_path);
        return;
    }

    match cli.coeff_type {
        CoeffType::Int => {
            let (instance, _) = match parser::parse_auto::<i64>(&contents, Some(&cli.file)) {
                Ok(res) => {
                    ::log::info!("parsed successfully (integer coefficients)");
                    res
                }
                Err(e) => {
                    ::log::error!("parse error: {e}");
                    process::exit(1);
                }
            };
            run_dispatch(instance, &cli, &logger);
        }
        CoeffType::Float => {
            let (instance, _) = match parser::parse_auto::<f64>(&contents, Some(&cli.file)) {
                Ok(res) => {
                    ::log::info!("parsed successfully (float coefficients)");
                    res
                }
                Err(e) => {
                    ::log::error!("parse error: {e}");
                    process::exit(1);
                }
            };
            run_dispatch(instance, &cli, &logger);
        }
        CoeffType::Auto => {
            // Try integer first; fall back to float.
            if let Ok((instance, _)) = parser::parse_auto::<i64>(&contents, Some(&cli.file)) {
                ::log::info!("parsed successfully (auto-detected integer coefficients)");
                run_dispatch(instance, &cli, &logger);
            } else {
                let (instance, _) = match parser::parse_auto::<f64>(&contents, Some(&cli.file)) {
                    Ok(res) => {
                        ::log::info!("parsed successfully (auto-detected float coefficients)");
                        res
                    }
                    Err(e) => {
                        ::log::error!("parse error: {e}");
                        process::exit(1);
                    }
                };
                run_dispatch(instance, &cli, &logger);
            }
        }
    }
}


const BANNER: &str = "\
██╗  ██╗██╗   ██╗███████╗███████╗
██║  ██║██║   ██║██╔════╝██╔════╝
███████║██║   ██║█████╗  ███████╗
██╔══██║██║   ██║██╔══╝  ╚════██║
██║  ██║╚██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝";

use owo_colors::OwoColorize;

const HUES_GRADIENT: [(u8, u8, u8); 6] = [
    (255,  89,  94), // red
    (255, 202,  58), // amber
    (138, 201,  38), // green
    ( 25, 130, 196), // blue
    (106,  76, 147), // violet
    (199,  74, 168), // magenta
];

pub fn print_banner() {
    println!();
    for (line, (r, g, b)) in BANNER.lines().zip(HUES_GRADIENT) {
        println!(" {}", line.truecolor(r, g, b));
    }
    println!(" {}", "─".repeat(33).dimmed());
    println!(
        " {}  v{}",
        "Higher-order Unconstrained Exact Solver".italic(),
        env!("CARGO_PKG_VERSION")
    );
    println!(" {}\n", "ZIB-AOPT · github.com/mlschicker/hues".dimmed());
}