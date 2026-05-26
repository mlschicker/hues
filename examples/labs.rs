//! Low Autocorrelation Binary Sequences (LABS) — example using the HypEx model API.
//!
//! The LABS problem seeks a binary sequence  s_0, …, s_{N-1}  with  s_i ∈ {-1, +1}
//! that minimises the *energy*
//!
//!   E(s) = ∑_{k=1}^{N-1}  C_k²
//!
//! where the autocorrelation at lag k is
//!
//!   C_k = ∑_{i=0}^{N-1-k}  s_i · s_{i+k}.
//!
//! Expanding C_k² gives a sum of degree-4 terms  s_i · s_{i+k} · s_j · s_{j+k},
//! which is exactly the kind of higher-order pseudo-Boolean objective that HUBO-TL
//! was designed for.
//!
//! Usage:
//!
//! ```text
//! cargo run --example labs -- 10 labs_10.hubo
//! cargo run --example labs -- 20              # writes to labs_20.hubo by default
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <N> [output.hubo]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("N must be a positive integer");
        process::exit(1);
    });

    let output = if args.len() >= 3 {
        args[2].clone()
    } else {
        format!("labs_{n}.hubo")
    };

    // Direct formulation:
    // E(s) = Σ_{k=1}^{N-1} ( Σ_{i=0}^{N-1-k} s_i s_{i+k} )²
    let model = HuboModel::spin(n)
        .with_meta("problem", "LABS")
        .with_meta("N", &n.to_string());

    let mut objective = model.expr_const(0.0);
    for k in 1..n {
        let mut autocorr = model.expr_const(0.0);
        for i in 0..(n - k) {
            autocorr = autocorr + model.expr_var(i) * model.expr_var(i + k);
        }
        objective = objective + autocorr.pow(2);
    }

    let instance = model.add_expr(objective).build();

    // Write to file
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("LABS instance written to {output}");
    eprintln!(
        "  N        = {n}\n  terms    = {}\n  offset   = {}\n  var_type = SPIN",
        instance.n_terms(), instance.offset
    );
}
