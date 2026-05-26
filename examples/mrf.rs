//! Higher-Order Markov Random Field HUBO instance generator.
//!
//! Generates a random Markov random field with linear, pairwise, and triple
//! clique potentials using random weights at fixed scales.
//!
//! Usage:
//!
//! ```text
//! cargo run --example mrf -- <n_vars> <output.hubo> [seed]
//! cargo run --example mrf -- 6 mrf_n6.hubo 1
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

    fn signed(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn scaled(&mut self, scale: f64) -> f64 {
        (self.signed() * scale * 1e12).round() / 1e12
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
    let pair_count = (n - 1).min((n / 2).max(3));
    let triple_count = (n - 2).min((n / 4).max(2));

    let mut model = HuboModel::spin(n)
        .with_meta("problem", "Higher-Order Markov Random Field")
        .with_meta("n_vars", &n.to_string())
        .with_meta("n_cliques", "3")
        .with_meta("seed", &seed.to_string());

    // Linear terms
    for i in 0..n {
        model = model.add_linear(i, rng.scaled(0.8));
    }

    // Pair terms
    for i in 0..pair_count {
        model = model.add_quadratic(i, i + 1, rng.scaled(0.9));
    }

    // Triple terms
    for i in 0..triple_count {
        model = model.add_term(&[i, i + 1, i + 2], rng.scaled(1.1));
    }

    let offset = -0.1 * n as f64 + 0.05 * pair_count as f64;
    let instance = model.with_offset(offset).build();

    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("MRF instance written to {output}");
    eprintln!(
        "  n_vars    = {n}\n  pair_count= {pair_count}\n  triple_count={triple_count}\n  terms     = {}\n  var_type  = SPIN",
        instance.n_terms()
    );
}
