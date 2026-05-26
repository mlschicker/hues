//! Higher-Order Hopfield Network HUBO instance generator.
//!
//! Builds a small associative-memory model with linear, pairwise, and triple
//! interactions derived from stored random patterns.
//!
//! Usage:
//!
//! ```text
//! cargo run --example hopfield -- <n_vars> <output.hubo> [seed]
//! cargo run --example hopfield -- 6 hopfield_n6.hubo 1
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn spin(&mut self) -> i32 {
        if self.next_u64() & 1 == 1 { 1 } else { -1 }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <n_vars> <output.hubo> [seed]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n_vars must be a positive integer");
        process::exit(1);
    });
    if n < 3 {
        eprintln!("n_vars must be at least 3");
        process::exit(1);
    }

    let output = args[2].clone();

    let seed: u64 = if args.len() >= 4 {
        args[3].parse().unwrap_or(42)
    } else {
        42
    };

    let mut rng = SplitMix64::new(seed);
    let n_patterns = 3usize;
    let patterns: Vec<Vec<i32>> = (0..n_patterns)
        .map(|_| (0..n).map(|_| rng.spin()).collect())
        .collect();

    let pair_count = (n - 1).min((n / 2).max(3));
    let triple_count = (n - 2).min((n / 4).max(2));

    let model = HuboModel::spin(n)
        .with_meta("problem", "Higher-Order Hopfield")
        .with_meta("n_vars", &n.to_string())
        .with_meta("n_patterns", &n_patterns.to_string())
        .with_meta("seed", &seed.to_string());

    let mut objective = model.expr_const(0.0);

    // Linear terms
    for i in 0..n {
        let coeff = -(patterns.iter().map(|p| p[i] as f64).sum::<f64>()) / n_patterns as f64;
        objective = objective + model.expr_const(coeff) * model.expr_var(i);
    }

    // Pair terms
    for i in 0..pair_count {
        let coeff = -(patterns.iter().map(|p| (p[i] * p[i + 1]) as f64).sum::<f64>())
            / n_patterns as f64;
        objective = objective + model.expr_const(coeff) * model.expr_var(i) * model.expr_var(i + 1);
    }

    // Triple terms
    for i in 0..triple_count {
        let coeff = -(patterns
            .iter()
            .map(|p| (p[i] * p[i + 1] * p[i + 2]) as f64)
            .sum::<f64>())
            / n_patterns as f64;
        objective = objective
            + model.expr_const(coeff)
                * model.expr_var(i)
                * model.expr_var(i + 1)
                * model.expr_var(i + 2);
    }

    let offset = -0.25 * n as f64 - 0.5 * n_patterns as f64;
    let instance = model.with_offset(offset).add_expr(objective).build();

    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Hopfield instance written to {output}");
    eprintln!(
        "  n_vars    = {n}\n  n_patterns= {n_patterns}\n  pair_count= {pair_count}\n  triple_count={triple_count}\n  terms     = {}\n  var_type  = SPIN",
        instance.n_terms()
    );
}
