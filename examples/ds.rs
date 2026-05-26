//! Dominating Set (PACE 2025 .gr) to HUBO converter.
//!
//! Reads a Dominating Set instance in the PACE 2025 format and writes an
//! equivalent binary HUBO objective:
//!
//!   min  ∑_{v=1}^n x_v  +  P * ∑_{v=1}^n ∏_{u∈N(v)∪{v}} (1 - x_u)
//!
//! where x_v in {0,1} indicates whether vertex v is in the dominating set.
//!
//! The product term is 1 iff vertex v is uncovered (neither v nor any
//! neighbor is chosen), and 0 otherwise. With P = n + 1, minimisers
//! correspond to minimum dominating sets.
//!
//! Input format (PACE 2025):
//! - lines starting with 'c' are comments
//! - first non-comment non-empty line: `p ds <n> <m>`
//! - then m non-comment non-empty lines, one edge per line: `u v` with u,v in [1..n]
//!
//! Usage:
//!
//! ```text
//! cargo run --example ds -- <input.gr> [output.hubo] [penalty]
//! cargo run --example ds -- instance.gr
//! cargo run --example ds -- instance.gr instance.hubo 250
//! ```

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use hues::model::HuboModel;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.gr> [output.hubo] [penalty]", args[0]);
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

    let instance = parse_pace_ds(&text).unwrap_or_else(|msg| {
        eprintln!("Input parse error: {msg}");
        process::exit(1);
    });

    println!(
        "Parsed Dominating Set instance: n_vertices={}, n_edges={}",
        instance.n_vertices,
        instance.edges.len()
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

    let hubo = dominating_set_to_hubo(&instance, penalty).unwrap_or_else(|msg| {
        eprintln!("Conversion error: {msg}");
        process::exit(1);
    });
    hubo.write_to_file(&output_path).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {e}", output_path.display());
        process::exit(1);
    });

    eprintln!("Dominating Set HUBO written to {}", output_path.display());
    eprintln!(
        "  vertices = {}\n  edges    = {}\n  terms    = {}\n  penalty  = {}\n  var_type = BIN",
        instance.n_vertices,
        instance.edges.len(),
        hubo.n_terms(),
        penalty
    );
}

struct DsInstance {
    n_vertices: usize,
    edges: Vec<(usize, usize)>, // 0-based vertex IDs
}

fn parse_pace_ds(input: &str) -> Result<DsInstance, String> {
    let mut header_seen = false;
    let mut n_vertices = 0usize;
    // let mut n_edges_expected = 0usize; // Unused variable removed
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        if !header_seen {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 || parts[0] != "p" || parts[1] != "ds" {
                return Err(format!(
                    "line {}: expected header `p ds <n> <m>`",
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

            // n_edges_expected = parts[3].parse::<usize>().map_err(|_| {
            //     format!(
            //         "line {}: invalid number of edges `{}`",
            //         line_no + 1,
            //         parts[3]
            //     )
            // })?;
            parts[3].parse::<usize>().map_err(|_| {
                format!(
                    "line {}: invalid number of edges `{}`",
                    line_no + 1,
                    parts[3]
                )
            })?;

            header_seen = true;
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        let u = parts[0]
            .parse::<usize>()
            .map_err(|_| format!("line {}: invalid vertex `{}`", line_no + 1, parts[0]))?;
        let v = parts[1]
            .parse::<usize>()
            .map_err(|_| format!("line {}: invalid vertex `{}`", line_no + 1, parts[1]))?;

        if u == 0 || u > n_vertices || v == 0 || v > n_vertices {
            return Err(format!(
                "line {}: vertices {} or {} out of range [1..{}]",
                line_no + 1,
                u,
                v,
                n_vertices
            ));
        }

        // Convert to 0-based
        let u0 = u - 1;
        let v0 = v - 1;

        if u0 != v0 {
            // Skip self-loops and duplicates
            let edge = if u0 < v0 { (u0, v0) } else { (v0, u0) };
            if !seen_edges.contains(&edge) {
                seen_edges.insert(edge);
                edges.push(edge);
            }
        }
    }

    if !header_seen {
        return Err("missing header `p ds <n> <m>`".to_string());
    }

    Ok(DsInstance { n_vertices, edges })
}

fn dominating_set_to_hubo(
    instance: &DsInstance,
    penalty: f64,
) -> Result<hues::instance::HuboInstance<f64, hues::domain::Bin>, String> {
    let n = instance.n_vertices;

    // Build adjacency list
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &instance.edges {
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    let mut model = HuboModel::binary(n)
        .with_meta("problem", "DominatingSet")
        .with_meta("format", "PACE2025-DS")
        .with_meta("n_edges", &instance.edges.len().to_string())
        .with_meta("penalty", &penalty.to_string())
        .with_meta(
            "var_semantics",
            "y_v = 1 - x_v (1 means vertex not in dominating set)",
        )
        .with_offset(n as f64);

    // n - Σ y_v (equivalent to Σ x_v in original formulation)
    for v in 0..n {
        model.add_term_mut(&[v], -1.0);
    }

    // + P * Σ_v Π_{u in neighborhood(v)} y_u
    for v in 0..n {
        // Build neighborhood: v and all its neighbors
        let mut neighborhood = vec![v];
        neighborhood.extend(&neighbors[v]);
        neighborhood.sort_unstable();
        neighborhood.dedup();

        // Add product term directly: penalty * ∏_{u in neighborhood} y_u
        model.add_term_mut(&neighborhood, penalty);
    }

    Ok(model.build())
}

fn default_output_path(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    out.set_extension("hubo");
    out
}
