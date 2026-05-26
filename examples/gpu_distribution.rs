//! GPU objective distribution sampler.
//!
//! Usage:
//!   cargo run --example gpu_distribution -- <instance.hubo> [samples] [bins]

use std::env;
use std::fs;
use std::process;

use hues::parser;
use hues::util::gpu_sampling::sample_and_print_distribution_gpu;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <instance.hubo> [samples] [bins]\nExample: {} ./instances/labs/labs_20.hubo 200000 50",
            args[0], args[0]
        );
        process::exit(1);
    }

    let path = &args[1];
    let samples: u64 = if args.len() >= 3 {
        args[2].parse().unwrap_or_else(|_| {
            eprintln!("samples must be a positive integer");
            process::exit(1);
        })
    } else {
        100_000
    };

    let bins: usize = if args.len() >= 4 {
        args[3].parse().unwrap_or_else(|_| {
            eprintln!("bins must be a positive integer");
            process::exit(1);
        })
    } else {
        40
    };

    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        process::exit(1);
    });

    let (instance, _) = parser::parse::<f64>(&text).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        process::exit(1);
    });

    eprintln!(
        "sampling {} random assignments on GPU (chunked; n_vars={}, n_terms={})...",
        samples,
        instance.n_vars(),
        instance.n_terms()
    );

    let result = match instance {
        hues::instance::HuboInstanceEnum::Bin(i) => {
            sample_and_print_distribution_gpu(&i, samples, bins)
        }
        hues::instance::HuboInstanceEnum::Spin(i) => {
            sample_and_print_distribution_gpu(&i, samples, bins)
        }
    };
    if let Err(e) = result {
        eprintln!("GPU sampling failed: {e}");
        process::exit(1);
    }
}
