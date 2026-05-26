//! Random ±J spin glass generator.
//!
//! Generates a random higher-order spin glass Hamiltonian where couplings
//! are drawn from a bimodal ±J distribution (Edwards-Anderson model
//! generalised to k-body interactions).
//!
//!   H(s) = −Σ_{T} J_T · Π_{i∈T} s_i,   s_i ∈ {-1,+1}
//!
//! where J_T ∈ {+J, −J} with equal probability (frac_pos controls the ratio).
//!
//! The ±J structure creates balanced frustration: in any spin configuration,
//! roughly half the couplings are satisfied and half are frustrated. This
//! leads to an exponentially rough energy landscape with extensive degeneracy
//! at the ground state — a defining feature of spin glass behaviour.
//!
//! Unlike Gaussian p-spin (which uses real-valued couplings), ±J gives
//! a frustration density that is exactly controllable via frac_pos.
//! Setting frac_pos=0.5 gives the maximally frustrated EA model.
//!
//! Usage:
//!
//! ```text
//! cargo run --example spin_glass -- <n> <m> <k> [output.hubo] [seed] [J] [frac_pos]
//! cargo run --example spin_glass -- 20 60 2
//! cargo run --example spin_glass -- 30 100 3 sg_n30_m100_k3.hubo 42 1.0 0.5
//! ```

use std::collections::HashSet;
use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <n> <m> <k> [output.hubo] [seed] [J] [frac_pos]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| { eprintln!("n invalid"); process::exit(1) });
    let m: usize = args[2].parse().unwrap_or_else(|_| { eprintln!("m invalid"); process::exit(1) });
    let k: usize = args[3].parse().unwrap_or_else(|_| { eprintln!("k invalid"); process::exit(1) });
    let output = if args.len() >= 5 { args[4].clone() } else { format!("spin_glass_n{n}_m{m}_k{k}.hubo") };
    let seed: u64 = if args.len() >= 6 { args[5].parse().unwrap_or(42) } else { 42 };
    let j_val: f64 = if args.len() >= 7 { args[6].parse().unwrap_or(1.0) } else { 1.0 };
    let frac_pos: f64 = if args.len() >= 8 { args[7].parse().unwrap_or(0.5) } else { 0.5 };

    if k < 1 || k > n { eprintln!("k must satisfy 1 <= k <= n"); process::exit(1); }
    if frac_pos < 0.0 || frac_pos > 1.0 { eprintln!("frac_pos must be in [0,1]"); process::exit(1); }

    let max_terms = binomial(n, k);
    let m_actual = m.min(max_terms);
    if m > max_terms {
        eprintln!("warning: m={m} > C({n},{k})={max_terms}; capping to {max_terms}");
    }

    let mut rng = SimpleRng::new(seed);
    let mut model = HuboModel::spin(n)
        .with_meta("problem", "RandomSpinGlass")
        .with_meta("n_vars", &n.to_string())
        .with_meta("n_terms_requested", &m.to_string())
        .with_meta("degree", &k.to_string())
        .with_meta("J", &j_val.to_string())
        .with_meta("frac_pos", &frac_pos.to_string())
        .with_meta("seed", &seed.to_string());

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let max_attempts = m_actual * 20;
    let mut attempts = 0;

    while seen.len() < m_actual && attempts < max_attempts {
        let mut indices: Vec<usize> = Vec::with_capacity(k);
        while indices.len() < k {
            let idx = rng.next_u64() as usize % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices.sort_unstable();
        if seen.insert(indices.clone()) {
            // H = -Σ J_T Π s_i  →  coefficient = -J_T
            let coupling = if rng.uniform() < frac_pos { j_val } else { -j_val };
            model = model.add_term(&indices, -coupling);
        }
        attempts += 1;
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Random ±J spin glass written to {output}");
    eprintln!(
        "  n={n}  m={}  k={k}  J={j_val}  frac_pos={frac_pos}  seed={seed}",
        instance.n_terms()
    );
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k { result = result * (n - i) / (i + 1); }
    result
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
