//! LP relaxation lower bound with simplex warm-starting across BnB nodes.
//!
//! Builds a linear programme (LP) by introducing one product variable per
//! monomial in the instance (left-to-right chain) and enforcing McCormick
//! inequalities.  The LP structure is **fixed** across all BnB nodes (all n
//! original variables are always present as LP columns, with fixed variables
//! having lb = ub = value).  This lets us warm-start each child node from its
//! parent's optimal simplex basis via SCIP's LP interface (`SCIPlpiSetBase` +
//! `SCIPlpiSolveDual`), so child LPs typically converge in very few pivots.
//!
//! Binary  (x_i ∈ [0,1]):  w ≤ a,  w ≤ b,  w ≥ a+b−1
//! Spin    (x_i ∈ [−1,1]): w ≤ 1+a−b,  w ≤ 1−a+b,  w ≥ a+b−1,  w ≥ −a−b−1

use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::ptr;

use russcip::ffi;

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;
use crate::solver::bnb::Node;
use crate::term::Term;

// ── Public types ──────────────────────────────────────────────────────────────

/// Simplex basis stored in a BnB node for warm-starting the LP at child nodes.
///
/// `cstat[i]` and `rstat[r]` contain SCIP_BASESTAT values for each LP column
/// and row respectively, extracted via `SCIPlpiGetBase` after solving and
/// restored via `SCIPlpiSetBase` before the next solve.
#[derive(Debug, Clone)]
pub struct LpBasis {
    pub(crate) cstat: Vec<i32>,
    pub(crate) rstat: Vec<i32>,
}

/// Configuration for the LP relaxation lower bound.
#[derive(Debug, Clone)]
pub struct LpConfig {
    /// Maximum number of LP columns (n_vars + n_product_vars).
    /// The LP is skipped and the trivial bound is returned when exceeded.
    pub max_cols: usize,
}

impl Default for LpConfig {
    fn default() -> Self {
        Self { max_cols: 500 }
    }
}

/// LP relaxation lower bound with simplex warm-starting.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct LpBound(pub LpConfig);


// ── Public compute entry point ────────────────────────────────────────────────

/// Compute the LP relaxation lower bound for `node`.
///
/// If `node.lp_basis` holds a basis from the parent, dual simplex warm-starts
/// from it.  After solving, the new optimal basis is stored back in
/// `node.lp_basis` so its children can warm-start in turn.
pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &mut Node<C>,
    cfg: &LpConfig,
) -> C {
    let n = instance.n_vars();
    let var_type = V::VAR_TYPE;

    // ── Build fixed LP structure from original instance terms ─────────────────
    let (prod_prefixes, chain_steps) = build_product_structure::<C>(n, &instance.terms);
    let n_prod = prod_prefixes.len();
    let n_cols = n + n_prod;

    if n_cols > cfg.max_cols {
        return trivial_lb(instance, node);
    }

    let prod_col_idx: HashMap<&[usize], usize> = prod_prefixes
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_slice(), n + i))
        .collect();

    // ── Variable bounds (fixed vars get lb = ub = their value) ───────────────
    let (xlb, xub) = match var_type {
        VarType::Bin => (0.0_f64, 1.0_f64),
        VarType::Spin => (-1.0_f64, 1.0_f64),
    };

    let mut col_lbs = vec![xlb; n_cols];
    let mut col_ubs = vec![xub; n_cols];
    for i in 0..n {
        if node.fixed.assigned.contains(i) {
            let v = if node.fixed.values.contains(i) { xub } else { xlb };
            col_lbs[i] = v;
            col_ubs[i] = v;
        }
    }

    // ── Objective (original coefficients on x and product columns) ───────────
    let mut obj = vec![0.0_f64; n_cols];
    let mut offset = instance.offset.to_f64();
    for term in &instance.terms {
        let c = term.coeff.to_f64();
        match term.indices.len() {
            0 => offset += c,
            1 => obj[term.indices[0]] += c,
            _ => {
                // Use the sorted version of the term's indices as the key.
                let mut sorted = term.indices.clone();
                sorted.sort_unstable();
                if var_type == VarType::Bin {
                    sorted.dedup();
                }
                if let Some(&col) = prod_col_idx.get(sorted.as_slice()) {
                    obj[col] += c;
                }
            }
        }
    }

    // ── McCormick rows ────────────────────────────────────────────────────────
    let rows_per_step: usize = match var_type {
        VarType::Bin => 3,
        VarType::Spin => 4,
    };
    let n_rows = chain_steps.len() * rows_per_step;

    let row_lhs = vec![f64::NEG_INFINITY; n_rows];
    let mut row_rhs: Vec<f64> = Vec::with_capacity(n_rows);
    let mut row_beg: Vec<i32> = Vec::with_capacity(n_rows);
    let mut row_ind: Vec<i32> = Vec::new();
    let mut row_val: Vec<f64> = Vec::new();

    for &(w_col, a_col, b_col) in &chain_steps {
        let w = w_col as i32;
        let a = a_col as i32;
        let b = b_col as i32;
        match var_type {
            VarType::Bin => {
                // w ≤ a  →  w − a ≤ 0
                row_rhs.push(0.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a]);
                row_val.extend_from_slice(&[1.0, -1.0]);
                // w ≤ b  →  w − b ≤ 0
                row_rhs.push(0.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, b]);
                row_val.extend_from_slice(&[1.0, -1.0]);
                // w ≥ a+b−1  →  −w+a+b ≤ 1
                row_rhs.push(1.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a, b]);
                row_val.extend_from_slice(&[-1.0, 1.0, 1.0]);
            }
            VarType::Spin => {
                // w ≤ 1+a−b  →  w−a+b ≤ 1
                row_rhs.push(1.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a, b]);
                row_val.extend_from_slice(&[1.0, -1.0, 1.0]);
                // w ≤ 1−a+b  →  w+a−b ≤ 1
                row_rhs.push(1.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a, b]);
                row_val.extend_from_slice(&[1.0, 1.0, -1.0]);
                // w ≥ a+b−1  →  −w+a+b ≤ 1
                row_rhs.push(1.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a, b]);
                row_val.extend_from_slice(&[-1.0, 1.0, 1.0]);
                // w ≥ −a−b−1  →  −w−a−b ≤ 1
                row_rhs.push(1.0);
                row_beg.push(row_ind.len() as i32);
                row_ind.extend_from_slice(&[w, a, b]);
                row_val.extend_from_slice(&[-1.0, -1.0, -1.0]);
            }
        }
    }
    debug_assert_eq!(row_beg.len(), n_rows);

    // ── Solve ─────────────────────────────────────────────────────────────────
    let warm_basis = node
        .lb_warm_start
        .as_ref()
        .and_then(|ws| ws.downcast_ref::<LpBasis>());
    match solve_node_lp(
        n_cols,
        &col_lbs,
        &col_ubs,
        &obj,
        n_rows,
        &row_lhs,
        &row_rhs,
        &row_beg,
        &row_ind,
        &row_val,
        warm_basis,
    ) {
        Ok((lp_val, new_basis)) => {
            node.lb_warm_start = Some(std::sync::Arc::new(new_basis));
            C::from_f64_lb(lp_val + offset)
        }
        Err(e) => {
            if e.contains("infeasible") {
                // LP infeasibility proves the node is infeasible → prune.
                C::max_value()
            } else {
                log::debug!("LP bound failed: {e}");
                trivial_lb(instance, node)
            }
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn trivial_lb<C: Coeff, V: VarDomain>(instance: &HuboInstance<C, V>, node: &Node<C>) -> C {
    let mut lb = instance.offset.to_f64() + node.offset.to_f64();
    for t in node.term_status.iter().flatten() {
        let c = t.coeff.to_f64();
        lb += match V::VAR_TYPE {
            VarType::Bin => c.min(0.0),
            VarType::Spin => -c.abs(),
        };
    }
    C::from_f64_lb(lb)
}

/// Build product variable prefixes and McCormick chain steps from the original
/// instance terms.  All indices are sorted (and deduped for binary) so that
/// the resulting LP column ordering is deterministic and consistent across
/// every BnB node for the same instance.
///
/// Returns `(prefixes, chain_steps)` where:
/// - `prefixes[i]` is the sorted prefix for LP column `n + i`
/// - `chain_steps[j] = (w_col, a_col, b_col)` in LP column indices
fn build_product_structure<C: Coeff>(
    n: usize,
    terms: &[Term<C>],
) -> (Vec<Vec<usize>>, Vec<(usize, usize, usize)>) {
    let mut prefix_set: BTreeSet<Vec<usize>> = BTreeSet::new();

    for term in terms {
        if term.indices.len() < 2 {
            continue;
        }
        let mut sorted = term.indices.clone();
        sorted.sort_unstable();
        sorted.dedup(); // safe for both binary (x²=x) and spin (x²=1 folds to constant)
        if sorted.len() < 2 {
            continue;
        }
        for k in 2..=sorted.len() {
            prefix_set.insert(sorted[..k].to_vec());
        }
    }

    // Sort by length first, then lexicographically, so shorter prefixes come
    // before longer ones — ensuring the left-factor of each chain step is
    // always created before the step that needs it.
    let mut prefixes: Vec<Vec<usize>> = prefix_set.into_iter().collect();
    prefixes.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

    let prod_col_idx: HashMap<Vec<usize>, usize> = prefixes
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), n + i))
        .collect();

    let mut chain_steps: Vec<(usize, usize, usize)> = Vec::with_capacity(prefixes.len());
    for prefix in &prefixes {
        let k = prefix.len();
        let w_col = prod_col_idx[prefix];
        let a_col = if k == 2 {
            prefix[0]
        } else {
            prod_col_idx[&prefix[..k - 1].to_vec()]
        };
        let b_col = prefix[k - 1];
        chain_steps.push((w_col, a_col, b_col));
    }

    (prefixes, chain_steps)
}

fn lpi_call(code: ffi::SCIP_Retcode, op: &str) -> Result<(), String> {
    if code == ffi::SCIP_Retcode_SCIP_OKAY {
        Ok(())
    } else {
        Err(format!("{op} returned retcode {code}"))
    }
}

/// Build and solve the LP using SCIP's LP interface.
///
/// If `warm_basis` is provided and has the correct dimensions, it is loaded
/// via `SCIPlpiSetBase` and the dual simplex is used (handles bound changes
/// from parent fixings with very few pivots).  Otherwise primal simplex runs
/// from scratch.
///
/// Returns `(lp_objective_value, new_basis)` on success, or `Err(msg)`.
/// An error message containing "infeasible" means the LP is provably infeasible.
fn solve_node_lp(
    n_cols: usize,
    col_lbs: &[f64],
    col_ubs: &[f64],
    obj: &[f64],
    n_rows: usize,
    row_lhs: &[f64],
    row_rhs: &[f64],
    row_beg: &[i32],
    row_ind: &[i32],
    row_val: &[f64],
    warm_basis: Option<&LpBasis>,
) -> Result<(f64, LpBasis), String> {
    unsafe {
        let mut lpi: *mut ffi::SCIP_LPI = ptr::null_mut();
        let name = CString::new("node_lp").map_err(|e| e.to_string())?;
        lpi_call(
            ffi::SCIPlpiCreate(
                &mut lpi,
                ptr::null_mut(),
                name.as_ptr(),
                ffi::SCIP_Objsense_SCIP_OBJSENSE_MINIMIZE,
            ),
            "SCIPlpiCreate",
        )?;

        let result: Result<(f64, LpBasis), String> = (|| {
            // ── Add columns (no row entries; rows will carry the constraint data) ──
            lpi_call(
                ffi::SCIPlpiAddCols(
                    lpi,
                    n_cols as i32,
                    obj.as_ptr(),
                    col_lbs.as_ptr(),
                    col_ubs.as_ptr(),
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                ),
                "SCIPlpiAddCols",
            )?;

            // ── Add McCormick rows ────────────────────────────────────────────────
            if n_rows > 0 {
                lpi_call(
                    ffi::SCIPlpiAddRows(
                        lpi,
                        n_rows as i32,
                        row_lhs.as_ptr(),
                        row_rhs.as_ptr(),
                        ptr::null_mut(),
                        row_ind.len() as i32,
                        row_beg.as_ptr(),
                        row_ind.as_ptr(),
                        row_val.as_ptr(),
                    ),
                    "SCIPlpiAddRows",
                )?;
            }

            // ── Warm-start or cold-start ──────────────────────────────────────────
            let use_warm = warm_basis
                .is_some_and(|b| b.cstat.len() == n_cols && b.rstat.len() == n_rows);

            if use_warm {
                let basis = warm_basis.unwrap();
                let set_ok = ffi::SCIPlpiSetBase(
                    lpi,
                    basis.cstat.as_ptr(),
                    basis.rstat.as_ptr(),
                );
                if set_ok == ffi::SCIP_Retcode_SCIP_OKAY {
                    // Dual simplex repairs bound changes from the parent node.
                    let solve_ok = ffi::SCIPlpiSolveDual(lpi);
                    if solve_ok != ffi::SCIP_Retcode_SCIP_OKAY {
                        // Dual simplex failed; fall back to primal.
                        lpi_call(ffi::SCIPlpiSolvePrimal(lpi), "SCIPlpiSolvePrimal")?;
                    }
                } else {
                    lpi_call(ffi::SCIPlpiSolvePrimal(lpi), "SCIPlpiSolvePrimal")?;
                }
            } else {
                lpi_call(ffi::SCIPlpiSolvePrimal(lpi), "SCIPlpiSolvePrimal")?;
            }

            // ── Check solve status ────────────────────────────────────────────────
            if ffi::SCIPlpiIsPrimalInfeasible(lpi) != 0 {
                return Err("LP infeasible".into());
            }
            if ffi::SCIPlpiIsOptimal(lpi) == 0 {
                return Err("LP did not reach optimality".into());
            }

            // ── Extract objective value ───────────────────────────────────────────
            let mut lp_val = 0.0_f64;
            lpi_call(
                ffi::SCIPlpiGetObjval(lpi, &mut lp_val),
                "SCIPlpiGetObjval",
            )?;

            // ── Extract basis for children's warm-start ───────────────────────────
            let mut cstat = vec![ffi::SCIP_BaseStat_SCIP_BASESTAT_LOWER as i32; n_cols];
            // Allocate at least 1 element so as_mut_ptr() is never null.
            let mut rstat = vec![ffi::SCIP_BaseStat_SCIP_BASESTAT_BASIC as i32; n_rows.max(1)];

            let get_ok = ffi::SCIPlpiGetBase(
                lpi,
                cstat.as_mut_ptr(),
                if n_rows > 0 {
                    rstat.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
            );
            rstat.truncate(n_rows);

            let new_basis = if get_ok == ffi::SCIP_Retcode_SCIP_OKAY {
                LpBasis { cstat, rstat }
            } else {
                log::debug!("SCIPlpiGetBase failed; warm-start unavailable for children");
                LpBasis { cstat: vec![], rstat: vec![] }
            };

            Ok((lp_val, new_basis))
        })();

        let _ = ffi::SCIPlpiFree(&mut lpi);
        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::Bin, instance::HuboInstance};

    fn make_bin(n: usize, terms: Vec<(Vec<usize>, f64)>) -> HuboInstance<f64, Bin> {
        HuboInstance::new(
            n,
            0.0,
            terms
                .into_iter()
                .map(|(indices, coeff)| Term { indices, coeff })
                .collect(),
        )
    }

    #[test]
    fn lp_bound_never_exceeds_optimum() {
        // min −x0·x1  s.t. x0,x1 ∈ {0,1}  → optimum = −1
        let inst = make_bin(2, vec![(vec![0, 1], -1.0)]);
        let mut node = crate::solver::bnb::Node::root(
            std::sync::Arc::new(inst.clone()),
            0.0_f64,
        );
        let lb = compute(&inst, &mut node, &LpConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ -1");
    }

    #[test]
    fn lp_bound_warm_starts_on_child() {
        // Solve root LP and verify basis is stored.
        let inst = make_bin(
            2,
            vec![(vec![0, 1], 1.0), (vec![0], -1.0), (vec![1], -1.0)],
        );
        let arc = std::sync::Arc::new(inst.clone());
        let mut root = crate::solver::bnb::Node::root(arc.clone(), 0.0_f64);
        let _lb_root = compute(&inst, &mut root, &LpConfig::default());

        assert!(
            root.lb_warm_start
                .as_ref()
                .and_then(|ws| ws.downcast_ref::<LpBasis>())
                .is_some(),
            "basis should be stored after root solve"
        );

        // Child inherits the basis.
        let child = root.child(&arc, 0, true);
        assert!(
            child
                .lb_warm_start
                .as_ref()
                .and_then(|ws| ws.downcast_ref::<LpBasis>())
                .is_some(),
            "child should inherit parent LP basis"
        );
    }

    #[test]
    fn lp_bound_is_tighter_than_trivial() {
        // min x0·x1 − x0 − x1: trivial = −2, LP optimum = −1
        let inst = make_bin(
            2,
            vec![(vec![0, 1], 1.0), (vec![0], -1.0), (vec![1], -1.0)],
        );
        let mut node = crate::solver::bnb::Node::root(
            std::sync::Arc::new(inst.clone()),
            0.0_f64,
        );
        let lb = compute(&inst, &mut node, &LpConfig::default());
        assert!(lb >= -2.0 - 1e-6, "LP must be ≥ trivial bound −2");
        assert!(lb > -2.0 + 0.5, "LP should be tighter than trivial");
        assert!(lb <= -1.0 + 1e-6, "LP must not exceed optimum −1");
    }
}
