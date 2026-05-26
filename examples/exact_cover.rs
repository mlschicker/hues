//! Exact Cover HUBO generator.
//!
//! Generates a random exact cover instance and writes the HUBO formulation.
//! Both the raw instance (.ec) and the HUBO model (.hubo) are written.
//!
//! Problem: Given universe U = {0,...,n-1} and m sets S_0,...,S_{m-1} ⊆ U,
//! find a minimum sub-collection that partitions U (each element covered
//! exactly once):
//!
//!   min   Σ_i y_i
//!   s.t.  Σ_{i: j∈S_i} y_i = 1   for all j ∈ U
//!         y_i ∈ {0,1}
//!
//! HUBO encoding (m binary variables y_i, one per set):
//!
//!   H(y) = Σ_i y_i  +  P · Σ_{j∈U} (Σ_{i: j∈S_i} y_i − 1)²
//!
//! Expanding each constraint (using y²=y for binary):
//!   linear:    -P · y_i  for each i covering j
//!   quadratic: +2P · y_{i1}·y_{i2}  for each pair (i1,i2) both covering j
//!
//! This is a QUBO (degree-2). Sparsity is controlled by the coverage density:
//! sparse instances (low density) have few interacting set pairs.
//!
//! Exact cover file format (.ec):
//!   p ec <n_elements> <n_sets>
//!   v1 v2 ... vk    (1-based element ids, one set per line)
//!
//! Usage:
//!
//! ```text
//! cargo run --example exact_cover -- <n> <m> [density] [base] [seed]
//! cargo run --example exact_cover -- 20 30
//! cargo run --example exact_cover -- 40 80 0.15 exact_cover_n40 42
//! ```

use std::collections::HashSet;
use std::env;
use std::fs;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <n> <m> [density] [base] [seed]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| { eprintln!("n invalid"); process::exit(1) });
    let m: usize = args[2].parse().unwrap_or_else(|_| { eprintln!("m invalid"); process::exit(1) });
    let density: f64 = if args.len() >= 4 { args[3].parse().unwrap_or(0.2) } else { 0.2 };
    let base = if args.len() >= 5 {
        args[4].clone()
    } else {
        format!("exact_cover_n{n}_m{m}")
    };
    let seed: u64 = if args.len() >= 6 { args[5].parse().unwrap_or(42) } else { 42 };

    if density <= 0.0 || density >= 1.0 {
        eprintln!("density must be in (0, 1)");
        process::exit(1);
    }

    let mut rng = SimpleRng::new(seed);

    // Generate random sets: each element j is in set i with prob density
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(); m];
    for j in 0..n {
        for i in 0..m {
            if rng.uniform() < density {
                sets[i].push(j);
            }
        }
        // Ensure every element is covered by at least one set
        if !sets.iter().any(|s| s.contains(&j)) {
            let fallback = rng.next_u64() as usize % m;
            sets[fallback].push(j);
        }
    }
    for s in &mut sets {
        s.sort_unstable();
        s.dedup();
    }

    // Write raw exact cover instance
    let ec_path = format!("{base}.ec");
    let mut ec = format!("p ec {n} {m}\n");
    for s in &sets {
        if s.is_empty() {
            ec.push('\n');
        } else {
            let line: Vec<String> = s.iter().map(|v| (v + 1).to_string()).collect();
            ec.push_str(&line.join(" "));
            ec.push('\n');
        }
    }
    fs::write(&ec_path, &ec).unwrap_or_else(|e| {
        eprintln!("Error writing {ec_path}: {e}"); process::exit(1);
    });

    // Build HUBO: m binary variables y_i (one per set)
    let penalty = (m + 1) as f64;

    // For each element j, find covering sets and build the constraint penalty
    // First index sets by covered element
    let mut covering: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, s) in sets.iter().enumerate() {
        for &j in s {
            covering[j].push(i);
        }
    }

    let mut model = HuboModel::binary(m)
        .with_meta("problem", "ExactCover")
        .with_meta("n_elements", &n.to_string())
        .with_meta("n_sets", &m.to_string())
        .with_meta("density", &format!("{density:.3}"))
        .with_meta("penalty", &(m + 1).to_string())
        .with_meta("seed", &seed.to_string());

    // Objective: Σ_i y_i
    for i in 0..m {
        model = model.add_term(&[i], 1.0);
    }

    // Penalty: P · Σ_j (Σ_{i∈cover(j)} y_i - 1)²
    for j in 0..n {
        let cov = &covering[j];
        // Linear part: -P · y_i for each i in cover(j)
        for &i in cov {
            model = model.add_term(&[i], -penalty);
        }
        // Quadratic part: +2P · y_{i1}·y_{i2}
        for ki in 0..cov.len() {
            for ki2 in (ki + 1)..cov.len() {
                model = model.add_term(&[cov[ki], cov[ki2]], 2.0 * penalty);
            }
        }
    }

    let hubo_path = format!("{base}.hubo");
    let instance = model.build();
    instance.write_to_file(&hubo_path).unwrap_or_else(|e| {
        eprintln!("Error writing {hubo_path}: {e}"); process::exit(1);
    });

    eprintln!("Exact cover instance written to {ec_path}");
    eprintln!("HUBO model written to {hubo_path}");
    eprintln!(
        "  n={n}  m={m}  density={density:.3}  n_terms={}  degree=2",
        instance.n_terms()
    );
}

struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
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
