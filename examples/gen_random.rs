//! Random HUBO instance generator.
//!
//! Generates a random polynomial objective with terms of mixed degree
//! (1 through `max_degree`).  Useful for stress-testing and benchmarking
//! solvers on unstructured instances.
//!
//! The generator creates `n_terms` terms, each with:
//! - a random degree $k$ sampled uniformly from $[1, \text{max\_degree}]$,
//! - $k$ distinct variable indices chosen uniformly from $[0, n)$,
//! - a coefficient sampled uniformly from $[-C_{\max}, C_{\max}]$.
//!
//! Usage:
//!
//! ```text
//! cargo run --example gen_random -- <n_vars> <n_terms> [max_degree] [var_type] [output.hubo] [seed]
//! cargo run --example gen_random -- 20 50
//! cargo run --example gen_random -- 30 100 4 spin random_30.hubo 42
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <n_vars> <n_terms> [max_degree] [bin|spin] [output.hubo] [seed]",
            args[0]
        );
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n_vars must be a positive integer");
        process::exit(1);
    });

    let m: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("n_terms must be a positive integer");
        process::exit(1);
    });

    let max_degree: usize = if args.len() >= 4 {
        args[3].parse().unwrap_or_else(|_| {
            eprintln!("max_degree must be a positive integer");
            process::exit(1);
        })
    } else {
        3
    };

    let var_type: &str = if args.len() >= 5 {
        &args[4]
    } else {
        "bin"
    };

    let output = if args.len() >= 6 {
        args[5].clone()
    } else {
        format!("random_n{n}_m{m}.hubo")
    };

    let seed: u64 = if args.len() >= 7 {
        args[6].parse().unwrap_or_else(|_| {
            eprintln!("seed must be a u64");
            process::exit(1);
        })
    } else {
        42
    };

    let mut rng = SimpleRng::new(seed);

    let coeff_max = 10.0;

    fn add_random_terms<V: hues::domain::VarDomain>(
        mut model: HuboModel<f64, V>,
        n: usize,
        m: usize,
        max_degree: usize,
        coeff_max: f64,
        rng: &mut SimpleRng,
    ) -> HuboModel<f64, V> {
        for _ in 0..m {
            let degree = 1 + (rng.next_u64() as usize % max_degree.min(n));
            let mut indices: Vec<usize> = Vec::with_capacity(degree);
            while indices.len() < degree {
                let idx = rng.next_u64() as usize % n;
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            indices.sort_unstable();
            let coeff = (rng.uniform() * 2.0 - 1.0) * coeff_max;
            model = model.add_term(&indices, coeff);
        }
        model
    }

    let (n_terms, offset_val) = match var_type {
        "spin" | "SPIN" => {
            let model = HuboModel::spin(n)
                .with_meta("problem", "Random")
                .with_meta("n_vars", &n.to_string())
                .with_meta("n_terms", &m.to_string())
                .with_meta("max_degree", &max_degree.to_string())
                .with_meta("seed", &seed.to_string());
            let model = add_random_terms(model, n, m, max_degree, coeff_max, &mut rng);
            let instance = model.build();
            instance.write_to_file(&output).unwrap_or_else(|e| {
                eprintln!("Error writing {output}: {e}");
                process::exit(1);
            });
            (instance.n_terms(), instance.offset)
        }
        _ => {
            let model = HuboModel::binary(n)
                .with_meta("problem", "Random")
                .with_meta("n_vars", &n.to_string())
                .with_meta("n_terms", &m.to_string())
                .with_meta("max_degree", &max_degree.to_string())
                .with_meta("seed", &seed.to_string());
            let model = add_random_terms(model, n, m, max_degree, coeff_max, &mut rng);
            let instance = model.build();
            instance.write_to_file(&output).unwrap_or_else(|e| {
                eprintln!("Error writing {output}: {e}");
                process::exit(1);
            });
            (instance.n_terms(), instance.offset)
        }
    };

    eprintln!("Random HUBO instance written to {output}");
    eprintln!(
        "  n_vars    = {n}\n  terms     = {}\n  max_deg   = {max_degree}\n  var_type  = {var_type}\n  offset    = {}",
        n_terms, offset_val
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

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
