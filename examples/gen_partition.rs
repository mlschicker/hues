//! Number Partitioning instance generator.
//!
//! Given a set of positive integers $w_1, \ldots, w_n$, the Number
//! Partitioning problem seeks a partition into two subsets whose sums are
//! as close as possible.  Using spin variables $s_i \in \{-1, +1\}$, this
//! is equivalent to **minimising**:
//!
//! $$
//!   f(s) = \left( \sum_{i=1}^{n} w_i \, s_i \right)^2
//!        = \sum_{i} \sum_{j} w_i \, w_j \, s_i \, s_j
//! $$
//!
//! A perfect partition yields $f = 0$.  The expanded objective has $n^2$
//! quadratic terms (including $n$ constant terms from $s_i^2 = 1$).
//!
//! This generator creates random weights uniformly sampled from $[1, W_{\max}]$.
//!
//! Usage:
//!
//! ```text
//! cargo run --example gen_partition -- <n_items> [max_weight] [output.hubo] [seed]
//! cargo run --example gen_partition -- 20
//! cargo run --example gen_partition -- 30 100 partition_30.hubo 42
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <n_items> [max_weight] [output.hubo] [seed]",
            args[0]
        );
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n_items must be a positive integer");
        process::exit(1);
    });

    let max_w: u64 = if args.len() >= 3 {
        args[2].parse().unwrap_or_else(|_| {
            eprintln!("max_weight must be a positive integer");
            process::exit(1);
        })
    } else {
        100
    };

    let output = if args.len() >= 4 {
        args[3].clone()
    } else {
        format!("partition_n{n}.hubo")
    };

    let seed: u64 = if args.len() >= 5 {
        args[4].parse().unwrap_or_else(|_| {
            eprintln!("seed must be a u64");
            process::exit(1);
        })
    } else {
        42
    };

    let mut rng = SimpleRng::new(seed);

    // Generate random weights in [1, max_w].
    let weights: Vec<f64> = (0..n)
        .map(|_| 1.0 + (rng.next_u64() % max_w) as f64)
        .collect();

    // f(s) = (∑ w_i s_i)² = ∑_i ∑_j  w_i w_j s_i s_j
    //
    // For i == j:  w_i² s_i² = w_i²  (constant, since s_i² = 1)
    // For i != j:  w_i w_j s_i s_j    (quadratic term)
    let constant: f64 = weights.iter().map(|w| w * w).sum();

    let mut model = HuboModel::spin(n)
        .with_offset(constant)
        .with_meta("problem", "NumberPartitioning")
        .with_meta("n_items", &n.to_string())
        .with_meta("max_weight", &max_w.to_string())
        .with_meta("seed", &seed.to_string());

    for (i, w) in weights.iter().enumerate() {
        println!("Var {} has weight {}", i, *w);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            // Coefficient is 2 * w_i * w_j (the 2 comes from symmetry: i,j and j,i).
            model = model.add_quadratic(i, j, 2.0 * weights[i] * weights[j]);
        }
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    let total_w: f64 = weights.iter().sum();
    eprintln!("Number Partitioning instance written to {output}");
    eprintln!(
        "  items    = {n}\n  Σw       = {total_w}\n  terms    = {}\n  offset   = {constant}\n  var_type = SPIN",
        instance.n_terms()
    );
}

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
}
