//! Densest subhypergraph HUBO generator.
//!
//! Generates a random k-uniform hypergraph and builds the HUBO for the
//! t-densest subhypergraph problem: find a subset S of exactly t vertices
//! maximising the number of hyperedges fully contained in S.
//!
//!   max  Σ_{e∈E} Π_{v∈e} x_v
//!   s.t. Σ_v x_v = t,  x_v ∈ {0,1}
//!
//! HUBO formulation (minimisation):
//!
//!   H(x) = −Σ_{e∈E} Π_{v∈e} x_v  +  P·(Σ_v x_v − t)²
//!
//! Penalty P = |E|+1.  Expanding (Σ x_v − t)² with x²=x:
//!   linear:    (1−2t)·x_v  per vertex
//!   quadratic: 2·x_u·x_v   per pair
//!
//! Degree of HUBO = k (the hyperedge order), making degree directly
//! controllable as a benchmark parameter.
//!
//! Both the raw hypergraph (.hgr) and HUBO model (.hubo) are written.
//!
//! Hypergraph format (.hgr):
//!   p dsg <n_vertices> <n_edges> <edge_order>
//!   v1 v2 ... vk    (1-based vertex ids, one hyperedge per line)
//!
//! Usage:
//!
//! ```text
//! cargo run --example densest_subhypergraph -- <n> <m> <k> <t> [base] [seed]
//! cargo run --example densest_subhypergraph -- 20 40 3 10
//! cargo run --example densest_subhypergraph -- 30 80 4 15 densest_n30 42
//! ```

use std::collections::HashSet;
use std::env;
use std::fs;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: {} <n> <m> <k> <t> [base] [seed]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n invalid");
        process::exit(1)
    });
    let m: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("m invalid");
        process::exit(1)
    });
    let k: usize = args[3].parse().unwrap_or_else(|_| {
        eprintln!("k invalid");
        process::exit(1)
    });
    let t: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("t invalid");
        process::exit(1)
    });

    if k < 2 || k > n {
        eprintln!("k must satisfy 2 <= k <= n");
        process::exit(1);
    }
    if t < 1 || t > n {
        eprintln!("t must satisfy 1 <= t <= n");
        process::exit(1);
    }

    let base = if args.len() >= 6 {
        args[5].clone()
    } else {
        format!("densest_n{n}_m{m}_k{k}_t{t}")
    };
    let seed: u64 = if args.len() >= 7 {
        args[6].parse().unwrap_or(42)
    } else {
        42
    };

    let mut rng = SimpleRng::new(seed);

    // Generate m distinct random k-subsets of [n]
    let edges = sample_k_subsets(n, m, k, &mut rng);
    let actual_m = edges.len();

    // Write raw hypergraph
    let hgr_path = format!("{base}.hgr");
    let mut hgr = format!("p dsg {n} {actual_m} {k}\n");
    for e in &edges {
        let line: Vec<String> = e.iter().map(|v| (v + 1).to_string()).collect();
        hgr.push_str(&line.join(" "));
        hgr.push('\n');
    }
    fs::write(&hgr_path, &hgr).unwrap_or_else(|e| {
        eprintln!("Error writing {hgr_path}: {e}");
        process::exit(1);
    });

    // Build HUBO
    let penalty = (actual_m + 1) as f64;
    let mut model = HuboModel::binary(n)
        .with_meta("problem", "DensestSubhypergraph")
        .with_meta("n_vertices", &n.to_string())
        .with_meta("n_edges", &actual_m.to_string())
        .with_meta("edge_order", &k.to_string())
        .with_meta("target_size", &t.to_string())
        .with_meta("penalty", &(actual_m + 1).to_string())
        .with_meta("seed", &seed.to_string());

    // Objective: -Σ_{e} Π_{v∈e} x_v
    for e in &edges {
        model = model.add_term(e, -1.0);
    }

    // Size penalty: P·(Σ x_v - t)²  →  P·(1-2t)·x_v + P·2·x_u·x_v
    let t_f = t as f64;
    for v in 0..n {
        model = model.add_term(&[v], penalty * (1.0 - 2.0 * t_f));
    }
    for u in 0..n {
        for v in (u + 1)..n {
            model = model.add_term(&[u, v], 2.0 * penalty);
        }
    }

    let hubo_path = format!("{base}.hubo");
    let instance = model.build();
    instance.write_to_file(&hubo_path).unwrap_or_else(|e| {
        eprintln!("Error writing {hubo_path}: {e}");
        process::exit(1);
    });

    eprintln!("Hypergraph written to {hgr_path}");
    eprintln!("HUBO model written to {hubo_path}");
    eprintln!(
        "  n={n}  m={actual_m}  k={k}  t={t}  n_terms={}",
        instance.n_terms()
    );
}

fn sample_k_subsets(n: usize, m: usize, k: usize, rng: &mut SimpleRng) -> Vec<Vec<usize>> {
    // Reservoir sampling of distinct k-subsets
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let max_attempts = m * 20;
    let mut attempts = 0;

    while seen.len() < m && attempts < max_attempts {
        let mut indices: Vec<usize> = Vec::with_capacity(k);
        while indices.len() < k {
            let idx = rng.next_u64() as usize % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices.sort_unstable();
        seen.insert(indices);
        attempts += 1;
    }

    let mut result: Vec<Vec<usize>> = seen.into_iter().collect();
    result.sort();
    result
}

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}
