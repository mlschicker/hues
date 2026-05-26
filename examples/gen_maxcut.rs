//! Max-Cut instance generator.
//!
//! Given a graph with weighted edges, the Max-Cut problem seeks a partition
//! of vertices into two sets that maximises the total weight of edges crossing
//! the partition.  Equivalently in SPIN form, **minimise**:
//!
//! $$
//!   f(s) = \sum_{(i,j) \in E} w_{ij} \, s_i \, s_j
//! $$
//!
//! When $s_i = s_j$ the term contributes $+w_{ij}$ (same side, penalised),
//! and when $s_i \ne s_j$ it contributes $-w_{ij}$ (cut edge, rewarded).
//!
//! This generator creates random Erdős–Rényi graphs $G(n, p)$ with uniform
//! edge weights in $[1, 10]$.
//!
//! Usage:
//!
//! ```text
//! cargo run --example gen_maxcut -- <n_vertices> <edge_prob> [output.hubo] [seed]
//! cargo run --example gen_maxcut -- 20 0.5
//! cargo run --example gen_maxcut -- 50 0.3 maxcut_50.hubo 42
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <n_vertices> <edge_prob> [output.hubo] [seed]",
            args[0]
        );
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n_vertices must be a positive integer");
        process::exit(1);
    });

    let p: f64 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("edge_prob must be a float in [0, 1]");
        process::exit(1);
    });

    let output = if args.len() >= 4 {
        args[3].clone()
    } else {
        format!("maxcut_n{n}_p{}.hubo", format!("{p:.1}").replace('.', ""))
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
    let mut model = HuboModel::spin(n)
        .with_meta("problem", "MaxCut")
        .with_meta("n_vertices", &n.to_string())
        .with_meta("edge_prob", &format!("{p}"))
        .with_meta("seed", &seed.to_string());

    let mut n_edges = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.uniform() < p {
                let weight = 1.0 + rng.uniform() * 9.0; // weight in [1, 10]
                model = model.add_quadratic(i, j, weight);
                n_edges += 1;
            }
        }
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Max-Cut instance written to {output}");
    eprintln!(
        "  vertices = {n}\n  edges    = {n_edges}\n  terms    = {}\n  var_type = SPIN",
        instance.n_terms()
    );
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
