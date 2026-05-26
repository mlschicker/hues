//! Hitting Set (PACE 2025 .hgr) to HUBO converter.
//!
//! Reads a Hitting Set instance in the PACE 2025 format and writes an
//! equivalent binary HUBO objective:
//!
//!   min  ∑_{v=1}^n x_v  +  P * ∑_{S in E} ∏_{v in S} (1 - x_v)
//!
//! where x_v in {0,1} indicates whether vertex v is chosen.
//!
//! The product term is 1 iff set S is uncovered, and 0 otherwise.
//! With P = n + 1, any infeasible solution is more expensive than every
//! feasible one, so minimisers correspond to minimum hitting sets.
//!
//! Implementation note:
//! We build an equivalent sparse objective over y_v = 1 - x_v:
//!
//!   min  n - ∑_{v=1}^n y_v + P * ∑_{S in E} ∏_{v in S} y_v
//!
//! This avoids expanding ∏(1 - x_v) into 2^|S| monomials.
//!
//! Input format (PACE 2025):
//! - lines starting with 'c' are comments
//! - first non-comment non-empty line: `p hs <n> <m>`
//! - then m non-comment non-empty lines, one set per line, vertices in [1..n]
//!
//! Usage:
//!
//! ```text
//! cargo run --example hs -- <input.hgr> [output.hubo] [penalty]
//! cargo run --example hs -- instance.hgr
//! cargo run --example hs -- instance.hgr instance.hubo 250
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.hgr> [output.hubo] [penalty]", args[0]);
        process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        default_output_path(input_path)
    };

    let text = fs::read_to_string(input_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {e}", input_path.display());
        process::exit(1);
    });

    let instance = parse_pace_hs(&text).unwrap_or_else(|msg| {
        eprintln!("Input parse error: {msg}");
        process::exit(1);
    });

    println!(
        "Parsed Hitting Set instance: n_vertices = {}, n_sets = {}",
        instance.n_vertices,
        instance.sets.len()
    );

    let default_penalty = (instance.n_vertices + 1) as f64;
    let penalty = if args.len() >= 4 {
        args[3].parse::<f64>().unwrap_or_else(|_| {
            eprintln!("penalty must be a number");
            process::exit(1);
        })
    } else {
        default_penalty
    };

    if penalty <= 0.0 {
        eprintln!("penalty must be > 0");
        process::exit(1);
    }

    let hubo = hitting_set_to_hubo(&instance, penalty).unwrap_or_else(|msg| {
        eprintln!("Conversion error: {msg}");
        process::exit(1);
    });

    println!(
        "Constructed HUBO instance: n_terms = {}, penalty = {}",
        hubo.n_terms(), penalty
    );

    hubo.write_to_file(&output_path).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {e}", output_path.display());
        process::exit(1);
    });

    eprintln!("Hitting Set HUBO written to {}", output_path.display());
    eprintln!(
        "  vertices = {}\n  sets     = {}\n  terms    = {}\n  penalty  = {}\n  var_type = BIN",
        instance.n_vertices,
        instance.sets.len(),
        hubo.n_terms(),
        penalty
    );
}

struct HsInstance {
    n_vertices: usize,
    sets: Vec<Vec<usize>>, // 0-based vertex IDs
}

fn parse_pace_hs(input: &str) -> Result<HsInstance, String> {
    let mut header_seen = false;
    let mut n_vertices = 0usize;
    let mut n_sets_expected = 0usize;
    let mut sets: Vec<Vec<usize>> = Vec::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        if !header_seen {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 || parts[0] != "p" || parts[1] != "hs" {
                return Err(format!(
                    "line {}: expected header `p hs <n> <m>`",
                    line_no + 1
                ));
            }

            n_vertices = parts[2].parse::<usize>().map_err(|_| {
                format!(
                    "line {}: invalid number of vertices `{}`",
                    line_no + 1,
                    parts[2]
                )
            })?;

            n_sets_expected = parts[3].parse::<usize>().map_err(|_| {
                format!(
                    "line {}: invalid number of sets `{}`",
                    line_no + 1,
                    parts[3]
                )
            })?;

            header_seen = true;
            continue;
        }

        let mut set: Vec<usize> = Vec::new();
        for token in line.split_whitespace() {
            let v = token
                .parse::<usize>()
                .map_err(|_| format!("line {}: invalid vertex `{token}`", line_no + 1))?;

            if v == 0 || v > n_vertices {
                return Err(format!(
                    "line {}: vertex {} out of range [1..{}]",
                    line_no + 1,
                    v,
                    n_vertices
                ));
            }
            set.push(v - 1);
        }

        if set.is_empty() {
            continue;
        }

        set.sort_unstable();
        set.dedup();
        sets.push(set);
    }

    if !header_seen {
        return Err("missing header `p hs <n> <m>`".to_string());
    }

    if sets.len() != n_sets_expected {
        return Err(format!(
            "set count mismatch: header says {}, parsed {} non-empty set lines",
            n_sets_expected,
            sets.len()
        ));
    }

    Ok(HsInstance { n_vertices, sets })
}

fn hitting_set_to_hubo(
    instance: &HsInstance,
    penalty: f64,
) -> Result<hues::instance::HuboInstance<f64, hues::domain::Bin>, String> {
    let n = instance.n_vertices;

    let mut model = HuboModel::binary(n)
        .with_meta("problem", "HittingSet")
        .with_meta("format", "PACE2025-HS")
        .with_meta("m_sets", &instance.sets.len().to_string())
        .with_meta("penalty", &penalty.to_string())
        .with_meta(
            "var_semantics",
            "y_v = 1 - x_v (1 means vertex not selected)",
        )
        .with_offset(n as f64);

    // n - Σ y_v
    for v in 0..n {
        model.add_term_mut(&[v], -1.0);
    }

    // + P * Σ_S Π_{v∈S} y_v
    for set in &instance.sets {
        model.add_term_mut(set, penalty);
    }

    Ok(model.build())
}

fn default_output_path(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    out.set_extension("hubo");
    out
}
