//! Binary Golay Pairs — HUSO generator.
//!
//! A binary Golay pair of length n is a pair of bipolar sequences (a, b),
//! each of length n with values in {+1, -1}, such that the sum of their
//! aperiodic autocorrelation functions vanishes at every non-zero lag:
//!
//!   C_a(k) + C_b(k) = 0  for k = 1, ..., n-1
//!
//! where  C_x(k) = Σ_{j=0}^{n-1-k} x_j · x_{j+k}.
//!
//! ## Formulation
//!
//! Spin variables: a_i, b_i ∈ {−1, +1} for i = 0..n-1.
//!
//! Variable layout:  a_i → index i,  b_i → index n+i.
//!
//! For each lag k ∈ {1, ..., n-1}:
//!
//!   corr_a(k) = Σ_{j=0}^{n-1-k} a_j · a_{j+k}
//!   corr_b(k) = Σ_{j=0}^{n-1-k} b_j · b_{j+k}
//!
//! Penalty Hamiltonian:
//!
//!   E = Σ_{k=1}^{n-1} ( corr_a(k) + corr_b(k) )²
//!
//! A global minimum of E = 0 corresponds to a valid Golay pair.
//! Known Golay lengths: 2, 4, 8, 10, 16, 20, 26, 32, ...
//!
//! ## Usage
//!
//! ```text
//! cargo run --example golay -- 2 golay_2.huso
//! cargo run --example golay -- 4              # writes golay_4.huso
//! ```

use std::env;
use std::fs;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <n> [output.huso]", args[0]);
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
        format!("golay_{n}.huso")
    };

    // Variable layout:
    //   a_i  →  index i      (i = 0..n)
    //   b_i  →  index n+i    (i = 0..n)
    let a = |i: usize| i;
    let b = |i: usize| n + i;

    let model = HuboModel::spin(2 * n)
        .with_meta("problem", "BinaryGolayPairs")
        .with_meta("n", &n.to_string());

    let mut obj = model.expr_const(0.0);

    // For each non-zero lag k, add the penalty ( corr_a(k) + corr_b(k) )².
    // The model automatically reduces s_i^2 = 1 when building the instance.
    for k in 1..n {
        // corr_a(k) = Σ_j  a_j · a_{j+k}
        let mut corr_a = model.expr_const(0.0);
        for j in 0..(n - k) {
            corr_a = corr_a + model.expr_var(a(j)) * model.expr_var(a(j + k));
        }

        // corr_b(k) = Σ_j  b_j · b_{j+k}
        let mut corr_b = model.expr_const(0.0);
        for j in 0..(n - k) {
            corr_b = corr_b + model.expr_var(b(j)) * model.expr_var(b(j + k));
        }

        let sum = corr_a + corr_b;
        obj = obj + sum.pow(2);
    }

    let instance = model.add_expr(obj).build();

    fs::write(&output, instance.to_huso_tl(None)).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Golay pair instance written to {output}");
    eprintln!(
        "  n        = {n}\n  vars     = {}\n  terms    = {}\n  var_type = SPIN",
        2 * n,
        instance.n_terms()
    );
}
