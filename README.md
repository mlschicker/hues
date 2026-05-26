# HUES — Higher-Order Unconstrained Exact Solver

```
██╗  ██╗██╗   ██╗███████╗███████╗
██║  ██║██║   ██║██╔════╝██╔════╝
███████║██║   ██║█████╗  ███████╗
██╔══██║██║   ██║██╔══╝  ╚════██║
██║  ██║╚██████╔╝███████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝
```

**HUES** (**H**igher-Order **U**nconstrained **E**xact **S**olver) solves **HUBO** (Higher-Order Unconstrained Binary Optimization) problems
expressed as sparse polynomial objectives over binary or spin variables:

$$
\min_{v \in D^n} f(v) \coloneqq C + \sum_{t=1}^{M} c_t \prod_{i \in S_t} v_i
$$

where $C \in \mathbb{R}$ is some offset, $D = \{0,1\}$ (binary) or $D = \{-1,+1\}$ (spin), and terms can be of
**arbitrary degree**.

## Features

- **Multiple solver backends** — choose the right tool for the job:

  | Backend | Flag | Description |
  |---------|------|-------------|
  | BnB | `--solver bnb` | Exact custom branch-and-bound (default) |
  | SCIP MC | `--solver scip-mc` | Exact via SCIP + McCormick linearisation |
  | SCIP NL | `--solver scip-nl` | Exact via SCIP + nonlinear objective |
  | SA | `--solver sa` | Simulated annealing with Boltzmann acceptance |
  | Tabu | `--solver tabu` | Tabu search with short-term memory |
  | Greedy | `--solver greedy` | Steepest-descent with restarts |
  | SAW | `--solver saw` | Self-avoiding walk heuristic |
  | Pool | `--solver pool` | Diverse solution-pool hybrid |
  | PT | `--solver pt` | Parallel tempering |

- **HUBO-TL file format** — a clean, human-readable term-list format with
  support for comments, metadata, and arbitrary-degree monomials
  (see [`FILE_FORMAT.md`](FILE_FORMAT.md)).
- **Programmatic model builder** — construct instances in Rust code via a
  fluent API (`HuboModel::binary(n).add_term(&[0,1], 2.0).build()`).
  Includes a symbolic expression DSL with arithmetic, variable negation, and
  whole-expression powers.
- **Kernelization** — root-level and node-level preprocessing reductions
  (roof dual / QPBO, dominance, coupling, symmetry breaking).
- **Warm-start** — provide an initial solution from a file (`-i`) to seed
  any solver.
- **Cutoff** — stop early when a solution with objective ≤ a threshold is
  found (`-c`).
- **Logging** — configurable verbosity with optional file output.

## Installation

Requires **Rust 2024 edition** (1.85+). SCIP is bundled automatically.

```bash
cargo build --release
```

The binary is at `target/release/hues`.
If SCIP is not installed on your system, running the binary directly can lead to problems. 

Therefore run an instance via
```bash
cargo run --release -- [arguments] /path/to/instance
```

## Quick Start

### From a file

```bash
# Solve with the default exact solver (BnB)
hues problem.hubo

# Solve with simulated annealing, 10 s time limit, write solution
hues problem.hubo --solver sa -t 10 -s solution.sol

# Solve with tabu search, stop if objective ≤ 42
hues problem.hubo --solver tabu -c 42

# Solve exactly with SCIP (McCormick linearisation)
hues problem.hubo --solver scip-mc -t 300

# Solve exactly with BnB, 4 threads, using the trwbp lower bound
hues problem.hubo --solver bnb --bnb-threads 4 --bnb-lb-method trwbp -t 60

# Kernelization scan (no solving)
hues problem.hubo --kernelization-scan

# Parse only (no solving)
hues problem.hubo --parse-only

# Convert HUBO to HUSO
hues problem.hubo --convert-to huso -o problem.huso

# Convert to JSON
hues problem.hubo --convert-to json -o problem.json
```

### Programmatic API

```rust
use hues::model::HuboModel;

// min  2 * x_0 * x_1  −  3 * x_2  +  1.5
let instance = HuboModel::binary(3)
    .with_offset(1.5)
    .add_term(&[0, 1], 2.0)
    .add_term(&[2], -3.0)
    .build();

let obj = instance.evaluate(&[1.0, 0.0, 1.0]);
assert_eq!(obj, 1.5 + 0.0 - 3.0);  // = −1.5

// Direct symbolic modeling with powers and variable negation:
// min (x0 + x1 - 1)^2 + 3 * (1 - x2)
let model = HuboModel::binary(3);
let expr = (model.expr_var(0) + model.expr_var(1) - model.expr_const(1.0)).pow(2)
  + model.expr_const(3.0) * model.expr_not_var(2);
let instance2 = model.add_expr(expr).build();
assert!((instance2.evaluate(&[1.0, 0.0, 0.0]) - 3.0).abs() < 1e-9);
```

## CLI Reference

### General options

| Flag | Description |
|------|-------------|
| `<file>` | Path to a HUBO-TL input file (or directory for `--kernelization-scan`) |
| `--solver <backend>` | Solver backend (default: `bnb`) |
| `--coeff-type <auto\|int\|float>` | Coefficient numeric type (default: `auto`) |
| `-t, --time-limit <secs>` | Time limit in seconds |
| `-c, --cutoff <value>` | Stop when objective ≤ value |
| `-s, --solution-file <path>` | Write solution to file |
| `-i, --initial-solution <path>` | Warm-start from a solution file |
| `-v, --log-level <level>` | Log verbosity: `trace`, `debug`, `info`, `warn`, `error` |
| `-l, --log-file <path>` | Write detailed log to file |
| `--parse-only` | Parse and print instance; do not solve |
| `--kernelization-scan` | Run kernelization only (no solving); accepts a directory |
| `--kernelization-scan-csv <path>` | Append kernelization stats to CSV (requires `--kernelization-scan`) |
| `--convert-to <hubo\|huso\|json>` | Convert objective domain/format and write output |
| `-o, --output <path>` | Output file path used with `--convert-to` |

### SCIP options

| Flag | Description |
|------|-------------|
| `--scip-node-limit <n>` | Max branch-and-bound nodes |
| `--scip-gap <ratio>` | Relative MIP gap limit (e.g. `0.01` = 1%) |
| `--scip-threads <n>` | Number of solver threads |
| `--scip-verbosity <0–5>` | SCIP output verbosity (default: 4) |

### BnB options

#### Core

| Flag | Description |
|------|-------------|
| `--config <path>` | TOML config file for advanced BnB parameters |
| `--bnb-node-limit <n>` | Max branch-and-bound nodes |
| `--bnb-threads <n>` | Parallel worker threads (default: 1 = serial) |
| `--bnb-seed <u64>` | RNG seed for warm-start heuristics (default: 0) |
| `--bnb-optimality-tol <f>` | Prune when `lb + tol ≥ ub` (default: 1e-5) |
| `--bnb-no-heuristic-warmstart` | Disable heuristic warm-starts before search |
| `--bnb-heuristic-time <secs>` | Per-heuristic time budget for warm-starts (default: 0.5) |
| `--bnb-no-kernelization` | Disable root-level kernelization |
| `--bnb-no-node-kernelization` | Disable node-level kernelization |
| `--bnb-bound-log-pct <f>` | Min bound improvement to log, as % of gap (default: 1.0) |
| `--bnb-log-every-nodes <n>` | Progress log every `<n>` explored nodes (0 disables; default: 5000) |
| `--bnb-stats-csv <path>` | Append one structured benchmark row per run to CSV |

#### Lower-bound method

| Flag | Description |
|------|-------------|
| `--bnb-lb-method <method>` | Lower-bounding oracle: `cheap`, `trwbp`, `hittingset`, `subgradient`, `lasserre`, `chordal-sdp`, `exact-lasserre` (default: `cheap`) |

#### Subgradient options (`--bnb-lb-method subgradient`)

| Flag | Description |
|------|-------------|
| `--bnb-subgrad-max-iter <n>` | Max ascent iterations (default: 64) |
| `--bnb-subgrad-step-size <β>` | Polyak relaxation factor β ∈ (0, 2] (default: 1.0) |
| `--bnb-subgrad-step-decay <d>` | Per-iteration decay of β ∈ (0, 1] (default: 1.0) |

#### TRWBP options (`--bnb-lb-method trwbp`)

| Flag | Description |
|------|-------------|
| `--bnb-trwbp-max-iter <n>` | Message-passing sweeps (default: 8) |
| `--bnb-trwbp-damping <d>` | Damping factor ∈ [0, 1) (default: 0.5) |

#### Hitting-set options (`--bnb-lb-method hittingset`)

| Flag | Description |
|------|-------------|
| `--bnb-hs-max-cores <n>` | Max unsat cores (default: 64) |
| `--bnb-hs-max-search-nodes <n>` | Max nodes for internal hitting-set solver (default: 50000) |

#### Lasserre options (`--bnb-lb-method lasserre`)

| Flag | Description |
|------|-------------|
| `--bnb-lasserre-order <d>` | Hierarchy order d (default: 1) |
| `--bnb-lasserre-max-vars <n>` | Only apply when unassigned variables ≤ n (default: 30) |

#### Exact Lasserre options (`--bnb-lb-method exact-lasserre`)

| Flag | Description |
|------|-------------|
| `--bnb-exact-lasserre-order <d>` | Hierarchy order d (default: 1) |
| `--bnb-exact-lasserre-max-vars <n>` | Max free variables (default: 20) |

#### Cluster options

| Flag | Description |
|------|-------------|
| `--bnb-cluster-max-vars <n>` | Max free variables per cluster for cluster-subgradient LB (default: 15) |

### SA options

| Flag | Description |
|------|-------------|
| `--sa-temp <T₀>` | Initial temperature (default: 10) |
| `--sa-final-temp <Tf>` | Final temperature (default: 1e-6) |
| `--sa-cooling <rate>` | Cooling factor per sweep (default: 0.9995) |
| `--sa-restarts <n>` | Independent restarts (default: 10) |
| `--sa-seed <u64>` | PRNG seed (omit for time-based) |

### Tabu options

| Flag | Description |
|------|-------------|
| `--tabu-tenure <k>` | Iterations a variable stays tabu (default: √n) |
| `--tabu-max-iter <n>` | Max iterations per restart (default: unbounded) |
| `--tabu-restarts <n>` | Independent restarts (default: 1) |
| `--tabu-seed <u64>` | PRNG seed (omit for time-based) |

### Greedy options

| Flag | Description |
|------|-------------|
| `--greedy-restarts <n>` | Independent restarts (default: 1) |
| `--greedy-max-flips <n>` | Max improving flips per restart (default: unbounded) |
| `--greedy-seed <u64>` | PRNG seed (omit for time-based) |

### SAW options

| Flag | Description |
|------|-------------|
| `--saw-n-walks <n>` | Number of independent walks (default: 1) |
| `--saw-max-steps <n>` | Max steps per walk (default: unbounded) |
| `--saw-local-search` | Apply steepest-descent polish at end of each segment |
| `--saw-seed <u64>` | PRNG seed (omit for time-based) |

### Pool options

| Flag | Description |
|------|-------------|
| `--pool-size <n>` | Max solutions in the pool (default: 16) |
| `--pool-init <n>` | Initial random solutions sampled (default: 32) |
| `--pool-max-iter <n>` | Max offspring iterations (default: unbounded) |
| `--pool-xor-max-flips <n>` | Max differing bits flipped by XOR move (default: 8) |
| `--pool-seed <u64>` | PRNG seed (omit for time-based) |

### Parallel Tempering options

| Flag | Description |
|------|-------------|
| `--pt-replicas <n>` | Replicas per run (default: 12) |
| `--pt-runs <n>` | Independent parallel runs (default: 8) |
| `--pt-sweeps <n>` | Sweeps per run (default: 10000) |
| `--pt-swap-interval <n>` | Attempt swaps every n sweeps (default: 5) |
| `--pt-t-min <T>` | Minimum temperature (cold replica, default: 0.1) |
| `--pt-t-max <T>` | Maximum temperature (hot replica, default: 10.0) |
| `--pt-adapt-interval <n>` | Adapt ladder every n sweeps; 0 disables (default: 500) |
| `--pt-target-accept-rate <r>` | Target swap acceptance rate for adaptation (default: 0.25) |
| `--pt-no-greedy-after-swap` | Disable greedy descent after accepted swaps |
| `--pt-seed <u64>` | PRNG seed (omit for time-based) |

## HUBO-TL File Format

Instance files use the **HUBO-TL** (Term List) format. Full specification in
[`FILE_FORMAT.md`](FILE_FORMAT.md). Example:

```text
HUBO 1
VAR_TYPE BIN
N 4
M 3
OFFSET 1.5
# 2* x_0 * x_1
0 1 2.0
# −3 * x_2
2 -3.0
# 0.5 * x_1 * x_2 * x_3
1 2 3 0.5
```

## Project Structure

```
src/
  main.rs               CLI entry point
  lib.rs                Crate root
  parser.rs             HUBO-TL parser + solution file parser
  model.rs              HuboModel builder API + evaluate()
  instance.rs           HuboInstance type
  term.rs               Term representation
  coeff.rs              Coeff trait (i64 / f64)
  domain.rs             Variable domain types (Bin / Spin)
  state.rs              Solution state
  fixes.rs              Fixed-variable tracking
  interrupt.rs          SIGINT handler
  chordal_sdp.rs        Chordal SDP relaxation
  lasserre.rs           Lasserre hierarchy
  heuristic/
    mod.rs              Shared types (Status, HeuristicResult, …)
    sa.rs               Simulated Annealing
    tabu.rs             Tabu Search
    greedy.rs           Greedy steepest-descent
    saw.rs              Self-Avoiding Walk
    pool.rs             Diverse solution-pool hybrid
    parallel_tempering.rs  Parallel Tempering
  solver/
    mod.rs              Re-exports + SolverConfig
    scip.rs             SCIP (MIP linearisation + nonlinear)
    bnb/
      mod.rs            BnB entry point
      solve.rs          Main solve loop
      serial.rs         Serial BnB
      parallel.rs       Parallel BnB
      types.rs          BnB types
      branching.rs      Branching strategies
      cutting_planes.rs Cutting plane generation
      probing.rs        Probing
      enumerate.rs      Enumeration utilities
      reporting.rs      Progress reporting
      util.rs           BnB utilities
      constraints/      Constraint types (cover, parity, lex-order)
  kernelization/
    mod.rs              Kernelization entry
    roof_dual.rs        Roof dual / QPBO
    dominance.rs        Dominance reductions
    coupling.rs         Coupling reductions
    symmetry.rs         Symmetry breaking
    util.rs / error.rs
  util/
    mod.rs              Re-exports
    error.rs            ParseError, ParseWarning types
    bitset.rs           Bitset utilities
    set_ops.rs          Set operations
examples/
  labs.rs               LABS (Low Autocorrelation Binary Sequences)
  hs.rs                 Hitting Set (PACE 2025 .hgr input)
  (and more …)
```

## Solver Details

### BnB (exact, default)

Performs direct branch-and-bound on the HUBO objective over the original
variable domain:

- BIN instances branch on $x_i \in \{0,1\}$.
- SPIN instances branch on $s_i \in \{-1,+1\}$.

Preprocessing reductions (kernelization) are applied at the root and
optionally at each node. Several lower-bounding oracles are available
(`--bnb-lb-method`), from a cheap bound to full Lasserre SDP relaxations.
Parallel search is supported via `--bnb-threads`.

Progress logs are printed in a fixed-width table including: current lower
bound, incumbent objective, explored nodes, unexplored frontier, pruned
nodes, optimality gap %, and elapsed time.

### SCIP (exact)

Linearises the polynomial objective into a Mixed-Integer Program.
`scip-mc` uses McCormick envelopes for binary products; `scip-nl` passes the
full nonlinear objective via SCIP's FFI. Spin variables are converted via
$s_i = 2x_i - 1$. Both guarantee global optimality given sufficient time.

### Simulated Annealing

Geometric cooling schedule with single-variable flips and efficient **delta
evaluation** — only terms containing the flipped variable are recomputed.
Accepts worsening moves with Boltzmann probability $e^{-\Delta/T}$.
Supports multi-restart.

### Tabu Search

Steepest-descent with a short-term memory that forbids recently flipped
variables for a configurable *tenure*. An **aspiration criterion** overrides
the tabu status when a flip leads to a new global best. Full neighbourhood
evaluation at each iteration.

### Greedy

Iterated steepest-descent: repeatedly flip the variable that gives the
greatest improvement until no improving move exists. Multi-restart from
random initial solutions.

### SAW (Self-Avoiding Walk)

Biased random walk that avoids recently visited variables. Optional
steepest-descent polish at the end of each segment.

### Pool

Maintains a diverse pool of solutions and generates offspring by XOR
recombination, refined with local search. Balances quality and diversity.

### Parallel Tempering

Runs a ladder of replicas at different temperatures in parallel, periodically
attempting swap moves between adjacent replicas. The temperature ladder adapts
to maintain a target acceptance rate.

## Examples

### LABS (Low Autocorrelation Binary Sequences)

Generate a LABS instance and solve it:

```bash
# Generate a LABS instance with N=20
cargo run --example labs -- 20

# Solve with SA
hues labs_20.hubo --solver sa --sa-restarts 10 -t 30

# Solve exactly with BnB
hues labs_20.hubo --solver bnb -t 300
```

### Hitting Set (PACE 2025 `.hgr` input)

Convert a PACE Hitting Set instance to HUBO-TL and solve it:

```bash
# Convert .hgr to .hubo (default penalty: n+1)
cargo run --example hs -- instances/hs/example.hgr

# Convert with explicit output path and penalty
cargo run --example hs -- instance.hgr instance.hubo 250

# Solve the generated HUBO
hues instance.hubo --solver bnb -t 60
```

The generated objective is:

$$
\min\ \sum_{v=1}^{n} x_v + P\sum_{S\in E}\prod_{v\in S}(1-x_v),\quad x_v\in\{0,1\}
$$

with $P=n+1$ by default, which enforces feasibility (all sets hit) and then
minimises the hitting-set size.

## Solution File Format

Solution files (written with `-s`) use a simple text format:

```text
# HUES solution file
STATUS completed
METHOD SA
OBJECTIVE -48.0
BEST_BOUND n/a
TIME 1.234567
ITERATIONS 500000
SOLUTION
s0 = -1
s1 = 1
...
```

These files can be loaded back as warm-start hints via `-i`.

## License

See [`Cargo.toml`](Cargo.toml) for package metadata.
