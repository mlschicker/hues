//! P-Spin instance generator.
//!
//! The p-spin model generates a random polynomial optimization problem where
//! all terms have exactly degree `p` (p-body interactions). This is useful for
//! studying the complexity of higher-order polynomial optimization.
//!
//! The objective is to minimize:
//!
//! $$
//!   f(s) = \sum_{i_1 < i_2 < ... < i_p} J_{i_1,i_2,...,i_p} \, s_{i_1} \, s_{i_2} \, ... \, s_{i_p}
//! $$
//!
//! with random coefficients $J$ sampled uniformly from $[-10, 10]$.
//!
//! Usage:
//!
//! ```text
//! cargo run --example pspin -- <n_vars> <p_order> [n_terms] [output.hubo] [seed]
//! cargo run --example pspin -- 20 3
//! cargo run --example pspin -- 30 3 100 pspin_30_3.hubo 42
//! cargo run --example pspin -- 50 2 500 pspin_50_2_dense.hubo 99
//! ```
//!
//! If `n_terms` is omitted, it defaults to min(C(n,p)/2, 500) for a balanced density.

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <n_vars> <p_order> [n_terms] [output.hubo] [seed]",
            args[0]
        );
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n_vars must be a positive integer");
        process::exit(1);
    });

    let p: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("p_order must be a positive integer");
        process::exit(1);
    });

    if p > n {
        eprintln!("p_order cannot be larger than n_vars");
        process::exit(1);
    }

    if p < 1 {
        eprintln!("p_order must be at least 1");
        process::exit(1);
    }

    // Maximum number of p-tuples
    let max_terms = binomial(n, p);

    // Default to using ~50% of possible terms, capped at 500
    let default_n_terms = (max_terms / 2).min(500);

    let n_terms: usize = if args.len() >= 4 {
        args[3].parse().unwrap_or_else(|_| {
            eprintln!("n_terms must be a positive integer");
            process::exit(1);
        })
    } else {
        default_n_terms
    };

    if n_terms > max_terms {
        eprintln!(
            "warning: n_terms ({n_terms}) exceeds C({n},{p}) = {max_terms}; capping to {max_terms}"
        );
    }

    let n_terms = n_terms.min(max_terms);

    let output = if args.len() >= 5 {
        args[4].clone()
    } else {
        format!("pspin_n{n}_p{p}_m{n_terms}.hubo")
    };

    let seed: u64 = if args.len() >= 6 {
        args[5].parse().unwrap_or_else(|_| {
            eprintln!("seed must be a u64");
            process::exit(1);
        })
    } else {
        42
    };

    let mut rng = SimpleRng::new(seed);
    let mut model = HuboModel::spin(n)
        .with_meta("problem", "P-Spin")
        .with_meta("n_vars", &n.to_string())
        .with_meta("p_order", &p.to_string())
        .with_meta("n_terms_requested", &n_terms.to_string())
        .with_meta("seed", &seed.to_string());

    // Generate n_terms random p-tuples with random coefficients
    let coeff_max = 10.0;
    let mut generated = 0usize;

    // Use rejection sampling to avoid duplicates
    let mut seen_terms: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    while generated < n_terms {
        // Sample p distinct variable indices
        let mut indices: Vec<usize> = Vec::with_capacity(p);
        while indices.len() < p {
            let idx = rng.next_u64() as usize % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices.sort_unstable();

        if !seen_terms.contains(&indices) {
            // Random coefficient in [-coeff_max, coeff_max]
            let coeff = (rng.uniform() * 2.0 - 1.0) * coeff_max;
            model = model.add_term(&indices, coeff);
            seen_terms.insert(indices);
            generated += 1;
        }
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("P-Spin instance written to {output}");
    eprintln!(
        "  n_vars   = {n}\n  p_order  = {p}\n  max_p_tuples = {max_terms}\n  n_terms  = {}\n  var_type = SPIN\n  offset   = {}",
        instance.n_terms(), instance.offset
    );
}

/// Compute binomial coefficient C(n, k)
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

// Minimal PRNG (splitmix64-based) to avoid external dependencies.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
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

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
