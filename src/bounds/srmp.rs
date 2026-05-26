//! Sequential Reweighted Message Passing (SRMP) lower bound for HUBO.
//!
//! Each active term becomes a higher-order factor with a full 2^k truth table.
//! SRMP performs sequential edge updates: for each (factor, variable) pair in
//! order it absorbs the current unary reparameterization into the factor,
//! extracts the min-marginal back to the unary, and normalises the factor to
//! zero minimum.  Alternating forward/backward passes give a monotonically
//! non-decreasing dual bound that equals the local-polytope LP optimum at
//! convergence.
//!
//! Factors whose arity exceeds `max_arity` are handled with the cheap bound.

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;

use super::Node;

/// Hard cap: 2^MAX_FACTOR_ARITY must fit comfortably in RAM per factor.
pub const MAX_FACTOR_ARITY: usize = 20;

#[derive(Debug, Clone, Copy)]
pub struct Srmp {
    pub max_iter: usize,
    pub optimality_tol: f64,
    /// Factors with more free variables than this fall back to the cheap bound.
    /// Capped internally at [`MAX_FACTOR_ARITY`].
    pub max_arity: usize,
}

impl Default for Srmp {
    fn default() -> Self {
        Self {
            max_iter: 20,
            optimality_tol: 1e-6,
            max_arity: 15,
        }
    }
}

// ---------------------------------------------------------------------------
// Factor representation
// ---------------------------------------------------------------------------

struct Factor {
    /// Sorted global variable indices for this factor.
    global_vars: Vec<usize>,
    /// Reparameterized cost table: 2^k entries.
    /// Bit j of the index = 1 means variable global_vars[j] takes its "high"
    /// state (1 for BIN, +1 for SPIN).
    table: Vec<f64>,
}

/// Build the initial factor table for a BIN HUBO term.
///
/// Only the all-ones entry (index `(1<<k)-1`) is non-zero.
fn init_factor_bin(vars: &[usize], coeff: f64) -> Factor {
    let k = vars.len();
    let size = 1 << k;
    let mut table = vec![0.0_f64; size];
    table[size - 1] = coeff;
    Factor {
        global_vars: vars.to_vec(),
        table,
    }
}

/// Build the initial factor table for a SPIN HUBO term.
///
/// For mask `m`: contribution = coeff * prod_j spin_j where
/// spin_j = +1 if bit j set, -1 otherwise.
/// Equivalently: coeff * (-1)^(number of zero bits in m within [0,k)).
fn init_factor_spin(vars: &[usize], coeff: f64) -> Factor {
    let k = vars.len();
    let size = 1 << k;
    let mut table = vec![0.0_f64; size];
    for mask in 0..size {
        let n_neg = k - mask.count_ones() as usize;
        table[mask] = if n_neg.is_multiple_of(2) { coeff } else { -coeff };
    }
    Factor {
        global_vars: vars.to_vec(),
        table,
    }
}

// ---------------------------------------------------------------------------
// Core SRMP update
// ---------------------------------------------------------------------------

/// Perform one SRMP edge update for the variable at local index `local_i`
/// inside `factor`.
///
/// 1. Absorb `unary[gi]` into the factor table, clear `unary[gi]`.
/// 2. Compute min-marginals h[0] and h[1] over the complementary variables.
/// 3. Subtract h from the factor (normalise to zero min-marginal).
/// 4. Add h to `unary[gi]`.
///
/// The dual bound is non-decreasing after each call.
#[inline]
fn update_edge(factor: &mut Factor, unary: &mut [[f64; 2]], local_i: usize) {
    let k = factor.global_vars.len();
    let gi = factor.global_vars[local_i];
    let size = 1usize << k;

    let u0 = unary[gi][0];
    let u1 = unary[gi][1];
    unary[gi] = [0.0, 0.0];

    let mut h = [f64::INFINITY; 2];

    // Combined absorb + min-marginal pass.
    for mask in 0..size {
        let xi = (mask >> local_i) & 1;
        let v = factor.table[mask] + if xi == 0 { u0 } else { u1 };
        factor.table[mask] = v;
        if v < h[xi] {
            h[xi] = v;
        }
    }

    // Normalise and extract.
    for mask in 0..size {
        let xi = (mask >> local_i) & 1;
        factor.table[mask] -= h[xi];
    }
    unary[gi][0] += h[0];
    unary[gi][1] += h[1];
}

// ---------------------------------------------------------------------------
// Dual bound
// ---------------------------------------------------------------------------

/// Compute the current dual bound.
///
/// After any edge update `min(factor.table) == 0`, so factor contributions
/// are zero once at least one variable has been processed.  We include them
/// anyway for correctness before the first pass.
fn dual_bound(factors: &[Factor], unary: &[[f64; 2]], base: f64) -> f64 {
    let mut d = base;
    for f in factors {
        d += f.table.iter().copied().fold(f64::INFINITY, f64::min);
    }
    for u in unary {
        d += u[0].min(u[1]);
    }
    d
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

fn srmp_lb<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &Srmp,
    incumbent_ub: Option<C>,
) -> f64 {
    let n = instance.n_vars();
    let var_type = V::VAR_TYPE;
    let max_arity = cfg.max_arity.min(MAX_FACTOR_ARITY);

    // Base constant: instance offset + node offset.
    let mut base = instance.offset.to_f64() + node.offset.to_f64();

    // Collect factors; fall back to cheap bound for oversized terms.
    let mut factors: Vec<Factor> = Vec::new();

    for ts in node.term_status.iter().flatten() {
        let coeff = ts.coeff.to_f64();
        if coeff == 0.0 {
            continue;
        }
        if ts.free_variables.is_empty() {
            base += coeff;
            continue;
        }
        let k = ts.free_variables.len();
        if k > max_arity {
            // Cheap bound contribution for this term.
            base += match var_type {
                VarType::Bin => coeff.min(0.0),
                VarType::Spin => -coeff.abs(),
            };
            continue;
        }
        factors.push(match var_type {
            VarType::Bin => init_factor_bin(&ts.free_variables, coeff),
            VarType::Spin => init_factor_spin(&ts.free_variables, coeff),
        });
    }

    if factors.is_empty() {
        log::info!("SRMP: no active terms, returning base={base}");
        return base;
    }

    let mut unary = vec![[0.0_f64; 2]; n];
    let mut best = dual_bound(&factors, &unary, base);
    let ub = incumbent_ub.map(|u| u.to_f64());

    for _ in 0..cfg.max_iter.max(1) {
        // Forward pass.
        for fi in 0..factors.len() {
            for li in 0..factors[fi].global_vars.len() {
                update_edge(&mut factors[fi], &mut unary, li);
            }
        }
        let d = dual_bound(&factors, &unary, base);
        if d > best {
            best = d;
        }
        if ub.is_some_and(|u| best >= u - cfg.optimality_tol) {
            break;
        }

        // Backward pass.
        for fi in (0..factors.len()).rev() {
            for li in (0..factors[fi].global_vars.len()).rev() {
                update_edge(&mut factors[fi], &mut unary, li);
            }
        }
        let d = dual_bound(&factors, &unary, base);
        if d > best {
            best = d;
        }
        if ub.is_some_and(|u| best >= u - cfg.optimality_tol) {
            break;
        }
    }

    best
}

/// Compute the SRMP lower bound for a BnB node.
pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &Srmp,
    incumbent_ub: Option<C>,
) -> C {
    let lb = srmp_lb(instance, node, cfg, incumbent_ub);
    let eps = 1e-9 * (1.0 + lb.abs());
    C::from_f64_lb(lb - eps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::bounds::cheap;
    use crate::domain::{Bin, Spin};
    use crate::instance::HuboInstance;
    use crate::solver::bnb::Node;
    use crate::term::Term;

    fn make_bin(n: usize, terms: Vec<(Vec<usize>, f64)>) -> Arc<HuboInstance<f64, Bin>> {
        let terms = terms
            .into_iter()
            .map(|(idx, c)| Term {
                indices: idx,
                coeff: c,
            })
            .collect();
        Arc::new(HuboInstance::new(n, 0.0, terms))
    }

    fn make_spin(n: usize, terms: Vec<(Vec<usize>, f64)>) -> Arc<HuboInstance<f64, Spin>> {
        let terms = terms
            .into_iter()
            .map(|(idx, c)| Term {
                indices: idx,
                coeff: c,
            })
            .collect();
        Arc::new(HuboInstance::new(n, 0.0, terms))
    }

    fn brute_min_bin(inst: &HuboInstance<f64, Bin>) -> f64 {
        let n = inst.n_vars();
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << n) {
            let mut v = inst.offset;
            for t in &inst.terms {
                if t.indices.iter().all(|&i| (mask >> i) & 1 == 1) {
                    v += t.coeff;
                }
            }
            if v < best {
                best = v;
            }
        }
        best
    }

    fn brute_min_spin(inst: &HuboInstance<f64, Spin>) -> f64 {
        let n = inst.n_vars();
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << n) {
            let mut v = inst.offset;
            for t in &inst.terms {
                let prod: f64 = t
                    .indices
                    .iter()
                    .map(|&i| if (mask >> i) & 1 == 1 { 1.0 } else { -1.0 })
                    .product();
                v += t.coeff * prod;
            }
            if v < best {
                best = v;
            }
        }
        best
    }

    #[test]
    fn srmp_bin_never_exceeds_optimum() {
        let inst = make_bin(
            6,
            vec![
                (vec![0, 1, 2], -4.0),
                (vec![1, 2, 3], 3.0),
                (vec![3, 4, 5], -2.0),
                (vec![0, 3], 1.5),
                (vec![2, 4], -1.0),
                (vec![1, 5], 2.0),
            ],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = Srmp {
            max_iter: 50,
            ..Default::default()
        };
        let lb = compute(inst.as_ref(), &node, &cfg, None);
        let opt = brute_min_bin(inst.as_ref());
        assert!(
            lb.to_f64() <= opt + 1e-6,
            "lb={} must not exceed opt={}",
            lb,
            opt
        );
    }

    #[test]
    fn srmp_bin_at_least_as_good_as_cheap() {
        let inst = make_bin(
            5,
            vec![
                (vec![0, 1, 2], -3.0),
                (vec![2, 3, 4], -2.0),
                (vec![0, 4], 1.0),
                (vec![1, 3], -1.5),
            ],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = Srmp {
            max_iter: 30,
            ..Default::default()
        };

        let srmp_lb = compute(inst.as_ref(), &node, &cfg, None).to_f64();
        let cheap_lb = {
            let mut n2 = node.clone();
            cheap::compute(inst.as_ref(), &mut n2).to_f64()
        };
        // SRMP applies a tiny eps floor so allow a small tolerance here.
        assert!(
            srmp_lb >= cheap_lb - 1e-6,
            "SRMP lb={srmp_lb} should be >= cheap lb={cheap_lb}"
        );
    }

    #[test]
    fn srmp_spin_never_exceeds_optimum() {
        let inst = make_spin(
            5,
            vec![
                (vec![0, 1, 2], -2.0),
                (vec![1, 2, 3], 1.5),
                (vec![0, 3, 4], -1.0),
                (vec![2, 4], 0.5),
            ],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = Srmp {
            max_iter: 50,
            ..Default::default()
        };
        let lb = compute(inst.as_ref(), &node, &cfg, None);
        let opt = brute_min_spin(inst.as_ref());
        assert!(
            lb.to_f64() <= opt + 1e-6,
            "lb={} must not exceed opt={}",
            lb,
            opt
        );
    }

    #[test]
    fn srmp_bin_tight_on_separable_instance() {
        // All negative coefficients, each term on disjoint variables: LP is tight.
        let inst = make_bin(
            6,
            vec![(vec![0, 1], -3.0), (vec![2, 3], -2.0), (vec![4, 5], -1.0)],
        );
        let node = Node::root(Arc::clone(&inst), f64::NEG_INFINITY);
        let cfg = Srmp {
            max_iter: 10,
            ..Default::default()
        };
        let lb = compute(inst.as_ref(), &node, &cfg, None).to_f64();
        let opt = brute_min_bin(inst.as_ref());
        assert!(
            (lb - opt).abs() <= 1e-6,
            "SRMP should be tight on separable instance: lb={lb}, opt={opt}"
        );
    }
}
