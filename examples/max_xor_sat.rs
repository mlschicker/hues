//! Max-XOR-k-SAT HUSO generator.
//!
//! A Max-XOR-k-SAT instance has n spin variables s_i ∈ {-1,+1} and m random
//! k-ary XOR clauses. Each clause specifies k variables and a target parity.
//! The goal is to maximise the number of satisfied clauses.
//!
//! Spin encoding (s = 1 - 2x, s ∈ {-1,+1}):
//!   Clause c with variables {i_1,...,i_k} and parity b_c ∈ {-1,+1} is
//!   satisfied iff  b_c · Π_j s_{i_j} = +1.
//!
//!   Maximising satisfied clauses ⟺ minimising:
//!     H(s) = −Σ_c b_c · Π_{j∈c} s_j
//!
//! with coefficients J_c = −b_c / 2 (scaled so optimal H = −m/2 if all satisfied).
//!
//! This is a random k-body spin glass with ±1/2 couplings. Unlike p-spin,
//! the coupling values are exactly ±1/2 (not Gaussian), and the clause
//! density α = m/n controls the frustration level. Near the satisfiability
//! threshold α*, the instances are hardest.
//!
//! Usage:
//!
//! ```text
//! cargo run --example max_xor_sat -- <n> <m> <k> [output.hubo] [seed]
//! cargo run --example max_xor_sat -- 20 60 3
//! cargo run --example max_xor_sat -- 30 90 4 max_xor_n30_m90_k4.hubo 42
//! ```

use std::collections::HashMap;
use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <n> <m> <k> [output.hubo] [seed]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| { eprintln!("n invalid"); process::exit(1) });
    let m: usize = args[2].parse().unwrap_or_else(|_| { eprintln!("m invalid"); process::exit(1) });
    let k: usize = args[3].parse().unwrap_or_else(|_| { eprintln!("k invalid"); process::exit(1) });
    let output = if args.len() >= 5 { args[4].clone() } else { format!("max_xor_n{n}_m{m}_k{k}.hubo") };
    let seed: u64 = if args.len() >= 6 { args[5].parse().unwrap_or(42) } else { 42 };

    if k < 2 || k > n {
        eprintln!("k must satisfy 2 <= k <= n");
        process::exit(1);
    }

    let alpha = m as f64 / n as f64;
    let mut rng = SimpleRng::new(seed);

    // Sample m XOR clauses with replacement; accumulate coefficients per distinct k-tuple
    let mut term_map: HashMap<Vec<usize>, f64> = HashMap::new();

    for _ in 0..m {
        let mut indices: Vec<usize> = Vec::with_capacity(k);
        while indices.len() < k {
            let idx = rng.next_u64() as usize % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices.sort_unstable();
        // Parity b ∈ {-1,+1}: H coefficient = -b (integer ±1)
        // Satisfied clause contributes -1 to H; unsatisfied contributes +1.
        // Optimal value = -m when all clauses are satisfied.
        let b: f64 = if rng.next_u64() % 2 == 0 { 1.0 } else { -1.0 };
        *term_map.entry(indices).or_insert(0.0) += -b;
    }

    let mut model = HuboModel::spin(n)
        .with_meta("problem", "MaxXORSAT")
        .with_meta("n_vars", &n.to_string())
        .with_meta("n_clauses", &m.to_string())
        .with_meta("clause_width", &k.to_string())
        .with_meta("alpha", &format!("{alpha:.3}"))
        .with_meta("seed", &seed.to_string());

    for (indices, coeff) in &term_map {
        if coeff.abs() > 1e-12 {
            model = model.add_term(indices, *coeff);
        }
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Max-XOR-{k}-SAT instance written to {output}");
    eprintln!(
        "  n={n}  m={m}  k={k}  alpha={alpha:.2}  distinct_terms={}  seed={seed}",
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
}
