//! Costas Array — MIP solved directly with SCIP.
//!
//! A Costas array of order n is a permutation σ of {1,…,n} where for every
//! lag k ∈ {1,…,n-1} the n-k differences σ(i+k)−σ(i) are all distinct.
//!
//! Unlike the HUBO example (`costas.rs`), this formulation uses explicit
//! linear constraints — no big-M penalties, no file I/O.
//!
//! ## Variables
//!
//! x[i][j] ∈ {0,1}  for i,j ∈ {0,…,n-1}
//!   x[i][j] = 1  ⟺  σ(i) = j   (0-indexed)
//!
//! Auxiliary product variables (McCormick linearisation):
//!   w[(a,b)] ∈ {0,1}  represents x[a] · x[b]
//!     w ≤ x[a],  w ≤ x[b],  w ≥ x[a] + x[b] − 1
//!
//! ## Constraints
//!
//! Permutation:
//!   Σ_j x[i][j] = 1   ∀ i        (one value per position)
//!   Σ_i x[i][j] = 1   ∀ j        (each value used once)
//!
//! Costas (for every lag k ≥ 1 and column displacement d ≠ 0):
//!   Σ_{valid (i,j)} w[(i·n+j, (i+k)·n+(j+d))] ≤ 1
//!   (at most one pair at lag k has difference d)
//!
//! Objective: 0  (pure feasibility — find any valid Costas array)
//!
//! ## Usage
//!
//! ```text
//! cargo run --example costas_mip -- 7
//! ```

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::process;

use russcip::Variable;
use russcip::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = if args.len() >= 2 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("n must be a positive integer");
            process::exit(1);
        })
    } else {
        6
    };
    if n < 2 {
        eprintln!("n must be at least 2");
        process::exit(1);
    }

    let mut model = Model::new()
        .include_default_plugins()
        .set_display_verbosity(5)
        // .set_param(param, value)
        .create_prob("costas_mip")
        .minimize();

    // all_vars[i*n + j] = x[i][j]; product variables are appended after.
    let mut all_vars: Vec<Variable> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            all_vars.push(model.add_var(
                0.0,
                1.0,
                0.0,
                &format!("x_{i}_{j}"),
                russcip::VarType::Binary,
            ));
        }
    }

    // One value per position.
    for i in 0..n {
        let vars: Vec<&Variable> = (0..n).map(|j| &all_vars[i * n + j]).collect();
        model.add_cons(vars, &vec![1.0; n], 1.0, 1.0, &format!("row_{i}"));
    }

    // Each value used exactly once.
    for j in 0..n {
        let vars: Vec<&Variable> = (0..n).map(|i| &all_vars[i * n + j]).collect();
        model.add_cons(vars, &vec![1.0; n], 1.0, 1.0, &format!("col_{j}"));
    }

    // Costas constraints via McCormick linearisation.
    // w_cache: (flat index a, flat index b) → index of product var in all_vars.
    let mut w_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut aux_count = 0usize;

    for k in 1..n {
        for d in -(n as isize - 1)..=(n as isize - 1) {
            if d == 0 {
                continue;
            }

            // Collect indices of product vars for all valid (i, j) at this (k, d).
            let mut costas_w: Vec<usize> = Vec::new();

            for i in 0..(n - k) {
                for j in 0..n {
                    let jd = j as isize + d;
                    if jd < 0 || jd >= n as isize {
                        continue;
                    }
                    let jd = jd as usize;
                    let a = i * n + j;
                    let b = (i + k) * n + jd;
                    let key = (a.min(b), a.max(b));

                    if !w_cache.contains_key(&key) {
                        let w_idx = all_vars.len();
                        let name = format!("w{aux_count}");
                        aux_count += 1;

                        // w is a local; add McCormick constraints before moving it.
                        let w = model.add_var(0.0, 1.0, 0.0, &name, russcip::VarType::Binary);
                        // w ≤ x[a]
                        model.add_cons(
                            vec![&w, &all_vars[key.0]],
                            &[1.0, -1.0],
                            f64::NEG_INFINITY,
                            0.0,
                            &format!("{name}_le_a"),
                        );
                        // w ≤ x[b]
                        model.add_cons(
                            vec![&w, &all_vars[key.1]],
                            &[1.0, -1.0],
                            f64::NEG_INFINITY,
                            0.0,
                            &format!("{name}_le_b"),
                        );
                        // w ≥ x[a] + x[b] − 1
                        model.add_cons(
                            vec![&w, &all_vars[key.0], &all_vars[key.1]],
                            &[1.0, -1.0, -1.0],
                            -1.0,
                            f64::INFINITY,
                            &format!("{name}_ge"),
                        );
                        all_vars.push(w);
                        w_cache.insert(key, w_idx);
                    }

                    costas_w.push(*w_cache.get(&key).unwrap());
                }
            }

            if costas_w.is_empty() {
                continue;
            }

            let vars: Vec<&Variable> = costas_w.iter().map(|&i| &all_vars[i]).collect();
            model.add_cons(
                vars,
                &vec![1.0; costas_w.len()],
                f64::NEG_INFINITY,
                1.0,
                &format!("costas_k{k}_d{d}"),
            );
        }
    }

    eprintln!(
        "n={n}: {} x-vars, {} product vars — solving…",
        n * n,
        aux_count
    );

    let solved = model.solve();
    let status = solved.status();

    println!("Status: {status:?}");
    println!("Solving time: {:.3}s", solved.solving_time());
    println!("Nodes: {}", solved.n_nodes());

    if let Some(sol) = solved.best_sol() {
        let mut perm = vec![0usize; n];
        for i in 0..n {
            for j in 0..n {
                if sol.val(&all_vars[i * n + j]) > 0.5 {
                    perm[i] = j + 1;
                }
            }
        }
        println!("Costas array (1-indexed): {perm:?}");

        let ok = (1..n).all(|k| {
            let diffs: Vec<isize> = (0..(n - k))
                .map(|i| perm[i + k] as isize - perm[i] as isize)
                .collect();
            let unique: HashSet<_> = diffs.iter().copied().collect();
            unique.len() == diffs.len()
        });
        println!("Costas property verified: {ok}");
    } else {
        println!("No feasible solution found for n={n}.");
    }
}
