//! Costas Array — HUBO generator.
//!
//! A Costas array of order n is an n×n permutation matrix (exactly one mark per
//! row and column) such that all n(n-1)/2 displacement vectors between the n
//! marked cells are distinct.
//!
//! ## Formulation
//!
//! Variables: x_{i,j} ∈ {0,1}, where x_{i,j} = 1 means cell (row i, col j) is
//! marked.  Variable index: i·n + j.
//!
//! The penalty Hamiltonian is:
//!
//! ```text
//! E = P_rc  · Σ_i ( Σ_j x_{i,j} − 1 )²        (one mark per row)
//!   + P_rc  · Σ_j ( Σ_i x_{i,j} − 1 )²        (one mark per column)
//!   + P_cos · Σ_{k=1}^{n-1} Σ_{d≠0} ( c_{k,d}² − c_{k,d} )
//! ```
//!
//! where c_{k,d} = Σ_{i,j valid} x_{i,j} · x_{i+k, j+d} counts how many
//! "L-pieces" at lag k share column displacement d.  The Costas property
//! requires every such count to be ≤ 1; the penalty c·(c-1) = c²-c is zero
//! iff c ≤ 1 and positive otherwise (for non-negative integer c).
//!
//! A global minimum of E = 0 corresponds to a valid Costas array.  For n ≤ 5
//! the problem is tiny; useful benchmark sizes are n = 6–12.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example costas -- 6 costas_6.hubo
//! cargo run --example costas -- 8              # writes costas_8.hubo
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <n> [output.hubo]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n must be a positive integer");
        process::exit(1);
    });

    if n < 2 {
        eprintln!("n must be at least 2");
        process::exit(1);
    }

    let output = if args.len() >= 3 {
        args[2].clone()
    } else {
        format!("costas_{n}.hubo")
    };

    // Variable layout: (row i, col j) -> i*n + j
    let idx = |i: usize, j: usize| i * n + j;

    // Penalty weights.
    // P_rc must be large enough to enforce the permutation constraints over the
    // Costas terms.  Each row/column constraint has degree-2 terms with
    // coefficients up to n (from expansion of (Σ x - 1)²).  Each c_{k,d}²
    // term has degree-4 terms with coefficient 1.  We set P_rc > P_cos · n so
    // that a violated permutation constraint always costs more than any
    // combination of Costas violations.
    let p_cos: f64 = 1.0;
    let p_rc: f64 = p_cos * (n as f64 + 1.0) * (n as f64);

    let model = HuboModel::binary(n * n)
        .with_meta("problem", "CostasArray")
        .with_meta("n", &n.to_string());

    let mut obj = model.expr_const(0.0);

    // --- Row constraints: ( Σ_j x_{i,j} - 1 )² for each row i ---
    for i in 0..n {
        let mut row_sum = model.expr_const(0.0);
        for j in 0..n {
            row_sum = row_sum + model.expr_var(idx(i, j));
        }
        let penalty = row_sum - model.expr_const(1.0);
        obj = obj + penalty.pow(2) * model.expr_const(p_rc);
    }

    // --- Column constraints: ( Σ_i x_{i,j} - 1 )² for each col j ---
    for j in 0..n {
        let mut col_sum = model.expr_const(0.0);
        for i in 0..n {
            col_sum = col_sum + model.expr_var(idx(i, j));
        }
        let penalty = col_sum - model.expr_const(1.0);
        obj = obj + penalty.pow(2) * model.expr_const(p_rc);
    }

    // --- Costas constraint ---
    // For each lag k (1..n) and column displacement d (≠ 0), accumulate
    // c_{k,d} = Σ_{valid (i,j)} x_{i,j} · x_{i+k, j+d}
    // and add the penalty p_cos · ( c_{k,d}² − c_{k,d} ).
    for k in 1..n {
        let d_min = -(n as isize - 1);
        let d_max = n as isize - 1;
        for d in d_min..=d_max {
            if d == 0 {
                continue;
            }
            let mut c_kd = model.expr_const(0.0);
            for i in 0..(n - k) {
                for j in 0..n {
                    let jd = j as isize + d;
                    if jd < 0 || jd >= n as isize {
                        continue;
                    }
                    c_kd = c_kd
                        + model.expr_var(idx(i, j))
                            * model.expr_var(idx(i + k, jd as usize));
                }
            }
            // c² - c is zero when c ≤ 1 and positive when c ≥ 2
            let c_sq = c_kd.clone().pow(2);
            obj = obj + (c_sq - c_kd) * model.expr_const(p_cos);
        }
    }

    let instance = model.add_expr(obj).build();

    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Costas array instance written to {output}");
    eprintln!(
        "  n        = {n}\n  vars     = {}\n  terms    = {}\n  var_type = BIN",
        n * n,
        instance.n_terms()
    );
}
