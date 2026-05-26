//! Random degree-k HUBO generator.
//!
//! Generates a random binary HUBO instance with exactly m terms of degree k.
//! Coefficients are drawn uniformly from [-coeff_max, coeff_max].
//!
//!   f(x) = Σ_{random k-tuples T} J_T · Π_{i∈T} x_i,   x_i ∈ {0,1}
//!
//! Parameters n (variables), m (term count), and k (degree) are independent,
//! giving precise control over density and interaction order separately.
//! This is the key distinguishing feature from p-spin models, which fix the
//! ratio m/C(n,k) rather than controlling m and k directly.
//!
//! Usage:
//!
//! ```text
//! cargo run --example random_hubo -- <n> <m> <k> [output.hubo] [seed] [coeff_max]
//! cargo run --example random_hubo -- 20 60 3
//! cargo run --example random_hubo -- 50 200 4 random_hubo_n50.hubo 42 10.0
//! ```

use std::collections::HashSet;
use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <n> <m> <k> [output.hubo] [seed] [coeff_max]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| { eprintln!("n invalid"); process::exit(1) });
    let m: usize = args[2].parse().unwrap_or_else(|_| { eprintln!("m invalid"); process::exit(1) });
    let k: usize = args[3].parse().unwrap_or_else(|_| { eprintln!("k invalid"); process::exit(1) });
    let output = if args.len() >= 5 { args[4].clone() } else { format!("random_hubo_n{n}_m{m}_k{k}.hubo") };
    let seed: u64 = if args.len() >= 6 { args[5].parse().unwrap_or(42) } else { 42 };
    let coeff_max: f64 = if args.len() >= 7 { args[6].parse().unwrap_or(10.0) } else { 10.0 };

    if k < 1 || k > n {
        eprintln!("k must satisfy 1 <= k <= n");
        process::exit(1);
    }

    let max_terms = binomial(n, k);
    let m_actual = m.min(max_terms);
    if m > max_terms {
        eprintln!("warning: m={m} > C({n},{k})={max_terms}; capping to {max_terms}");
    }

    let mut rng = SimpleRng::new(seed);
    let mut model = HuboModel::binary(n)
        .with_meta("problem", "RandomHUBO")
        .with_meta("n_vars", &n.to_string())
        .with_meta("n_terms_requested", &m.to_string())
        .with_meta("degree", &k.to_string())
        .with_meta("coeff_max", &coeff_max.to_string())
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
            let coeff = (rng.uniform() * 2.0 - 1.0) * coeff_max;
            model = model.add_term(&indices, coeff);
        }
        attempts += 1;
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("Random HUBO written to {output}");
    eprintln!(
        "  n={n}  m={}  k={k}  coeff_max={coeff_max}  seed={seed}",
        instance.n_terms()
    );
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
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
