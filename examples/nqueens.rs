//! N-D-Queens HUBO instance generator.
//!
//! Builds an N-D-Queens instance where:
//! - `n` is the size of each axis,
//! - `d` is the number of dimensions,
//! - the board has `n^d` cells.
//!
//! We use binary variables `x_p` for each position `p` in the `d`-dimensional grid.
//! The model aims to place as many queens as possible while discouraging attacks.
//! Since HypEx minimizes, we use:
//!
//! `f(x) = -sum_p x_p + B * sum_{(p,q) attacking, p<q} x_p x_q`
//!
//! Minimizing `f` is equivalent to maximizing the number of queens, with
//! `attack_penalty = B` controlling how strongly attacking placements are discouraged.
//!
//! Two positions attack each other iff their coordinate differences are aligned
//! with a queen move in D dimensions: all non-zero absolute deltas are equal.
//!
//! Usage:
//!
//! ```text
//! cargo run --example nqueens -- <n> <d> [attack_penalty] [output.hubo]
//! cargo run --example nqueens -- 8 2
//! cargo run --example nqueens -- 4 3 2.0 n3d4.hubo
//! ```

use std::env;
use std::process;

use hues::model::HuboModel;

fn index_to_coords(mut idx: usize, n: usize, d: usize) -> Vec<usize> {
    let mut coords = vec![0usize; d];
    for i in (0..d).rev() {
        coords[i] = idx % n;
        idx /= n;
    }
    coords
}

fn attack_in_d_dimensions(a: &[usize], b: &[usize]) -> bool {
    let mut step: Option<usize> = None;
    for i in 0..a.len() {
        let delta = a[i].abs_diff(b[i]);
        if delta == 0 {
            continue;
        }
        match step {
            None => step = Some(delta),
            Some(s) if s == delta => {}
            Some(_) => return false,
        }
    }

    step.is_some()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <n> <d> [attack_penalty] [output.hubo]", args[0]);
        process::exit(1);
    }

    let n: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("n must be a positive integer");
        process::exit(1);
    });

    if n == 0 {
        eprintln!("n must be >= 1");
        process::exit(1);
    }

    let d: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("d must be a positive integer");
        process::exit(1);
    });

    if d == 0 {
        eprintln!("d must be >= 1");
        process::exit(1);
    }

    let attack_penalty: f64 = if args.len() >= 4 {
        args[3].parse().unwrap_or_else(|_| {
            eprintln!("attack_penalty must be a float");
            process::exit(1);
        })
    } else {
        2.0
    };

    let output = if args.len() >= 5 {
        args[4].clone()
    } else {
        format!("n{n}_d{d}_queens.hubo")
    };

    let n_vars = n.pow(d as u32);
    let mut model = HuboModel::binary(n_vars)
        .with_meta("problem", "N-D-Queens")
        .with_meta("n", &n.to_string())
        .with_meta("d", &d.to_string())
        .with_meta("attack_penalty", &attack_penalty.to_string())
        .with_meta("objective", "maximize_queen_count_with_attack_penalty");

    // Precompute coordinates for all cells.
    let coords: Vec<Vec<usize>> = (0..n_vars).map(|i| index_to_coords(i, n, d)).collect();

    // Reward each queen: minimizing -sum(x) maximizes queen count.
    for i in 0..n_vars {
        model = model.add_linear(i, -1.0);
    }

    // Penalize attacking pairs anywhere on the board.
    let mut n_attack_pairs = 0usize;
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            if attack_in_d_dimensions(&coords[i], &coords[j]) {
                model = model.add_quadratic(i, j, attack_penalty);
                n_attack_pairs += 1;
            }
        }
    }

    let instance = model.build();
    instance.write_to_file(&output).unwrap_or_else(|e| {
        eprintln!("Error writing {output}: {e}");
        process::exit(1);
    });

    eprintln!("N-D-Queens instance written to {output}");
    eprintln!(
        "  n_vars          = {n_vars}\n  board_size      = {n}^{d}\n  terms           = {}\n  attack_pairs    = {n_attack_pairs}\n  var_type        = BIN\n  queen_reward    = 1.0\n  attack_penalty  = {attack_penalty}\n  offset          = {}",
        instance.n_terms(), instance.offset
    );
}
