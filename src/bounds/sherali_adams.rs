//! Sherali–Adams LP lower bound for HUBO with an explicit separation routine.
//!
//! The Sherali–Adams (SA) hierarchy for 0-1 programmes lifts each product
//! x_S = Π_{i∈S} x_i by multiplying the bound constraints (x_i ≥ 0, x_i ≤ 1,
//! and their complements) by each other, producing McCormick-type inequalities:
//!
//! Binary step w = a · b (a,b ∈ [0,1]):
//!   w ≤ a,   w ≤ b,   w ≥ a + b − 1,   w ≥ 0
//!
//! Spin step w = a · b (a,b ∈ [−1,1]):
//!   w ≤ 1+a−b,  w ≤ 1−a+b,  w ≥ a+b−1,  w ≥ −a−b−1
//!
//! ## Hierarchy levels
//!
//! The `level` parameter controls which product variables and chain steps are
//! included, beyond the left-to-right chain that is always created to connect
//! each term's objective variable to the x variables:
//!
//! | Level | Products added per term t = [i₀,…,i_{k−1}]                       |
//! |-------|-------------------------------------------------------------------|
//! |   0   | Left-to-right chain only (original behaviour)                    |
//! |   1   | All C(k,2) pairs — every (iₚ,i_q) subset of t (SA-1 full)       |
//! |   2   | All pairs + all C(k,3) triples with all 3 McCormick orderings     |
//! |  ≥2   | Cross-term pairs from unions of overlapping terms also added     |
//!
//! Unlike the RLT LP (which adds ALL McCormick constraints upfront), the SA
//! approach here uses **lazy separation**: the LP starts with product variables
//! but no linking constraints.  After each solve the separation oracle scans
//! all registered chain steps and adds only violated inequalities.
//! This produces a sequence of increasingly tight LPs, each solved by SCIP.

use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::ptr;

use russcip::ffi;

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;
use crate::solver::bnb::Node;

/// Configuration for the Sherali–Adams LP lower bound.
#[derive(Debug, Clone)]
pub struct SheraliAdamsConfig {
    /// SA hierarchy level (see module doc table).
    /// Level 0: left-to-right chain only.
    /// Level 1: all within-term pairs (full SA-1).
    /// Level 2: all within-term pairs + all triples with all McCormick orderings,
    ///           plus cross-term pairs for overlapping terms (full SA-2 within terms).
    pub level: usize,
    /// Maximum number of separation rounds (solve → separate → add → re-solve).
    pub max_rounds: usize,
    /// Minimum absolute violation required to add a cut.
    pub violation_tol: f64,
    /// Skip computation if free variables exceed this.
    pub max_vars: usize,
}

impl Default for SheraliAdamsConfig {
    fn default() -> Self {
        Self {
            level: 2,
            max_rounds: 100,
            violation_tol: 1e-6,
            max_vars: 200,
        }
    }
}

/// Newtype used as the `LowerBound` impl.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct SheraliAdams(pub SheraliAdamsConfig);


// ── public API ────────────────────────────────────────────────────────────────

/// A single violated SA cut produced by the separation oracle.
#[derive(Debug, Clone)]
pub struct SaViolation {
    /// The product variable whose SA constraint is violated.
    pub prefix: Vec<usize>,
    /// The specific left factor for this McCormick step (determines the ordering).
    pub left: ChainFactor,
    /// The specific right x index for this McCormick step.
    pub right_x: usize,
    pub cut: SaCutKind,
    /// Magnitude of the violation (positive).
    pub violation: f64,
}

/// Which specific McCormick inequality is violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SaCutKind {
    /// w > a  (binary: w−a ≤ 0;  spin: w−a+b ≤ 1)
    WLeqA,
    /// w > b  (binary: w−b ≤ 0;  spin: w+a−b ≤ 1)
    WLeqB,
    /// w < a+b−1  (both: −w+a+b ≤ 1)
    WGeqAplusB,
    /// [spin only]  w > −a−b−1  (−w−a−b ≤ 1)
    WGeqMinusAMinusB,
}

/// Left factor in a product chain step.
#[derive(Debug, Clone)]
pub enum ChainFactor {
    X(usize),
    Prod(Vec<usize>),
}

/// Run the SA separation oracle on a given LP solution.
///
/// `x_vals`    – value of x[i] in the current LP solution (local var index)
/// `prod_vals` – map from sorted product prefix to LP value
/// `chain`     – list of (w_prefix, left_factor, right_x_idx) chain steps
/// `var_type`  – binary or spin domain
/// `tol`       – minimum violation to report
///
/// Returns all violated cuts sorted by descending violation magnitude.
pub fn separate_sa_cuts(
    x_vals: &[f64],
    prod_vals: &HashMap<Vec<usize>, f64>,
    chain: &[(Vec<usize>, ChainFactor, usize)],
    var_type: VarType,
    tol: f64,
) -> Vec<SaViolation> {
    let mut violations = Vec::new();

    for (prefix, left, right_x) in chain {
        let w_val = match prod_vals.get(prefix) {
            Some(&v) => v,
            None => continue,
        };
        let b_val = x_vals[*right_x];
        let a_val = match left {
            ChainFactor::X(i) => x_vals[*i],
            ChainFactor::Prod(p) => match prod_vals.get(p) {
                Some(&v) => v,
                None => continue,
            },
        };

        let cuts =
            sa_violations_for_step(var_type, w_val, a_val, b_val, prefix, left, *right_x, tol);
        violations.extend(cuts);
    }

    violations.sort_unstable_by(|a, b| b.violation.partial_cmp(&a.violation).unwrap());
    violations
}

/// Compute the SA lower bound for the given (possibly partial) assignment.
pub fn sherali_adams_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
    cfg: &SheraliAdamsConfig,
) -> f64 {
    let var_type = V::VAR_TYPE;

    let mut g2l = vec![usize::MAX; instance.n_vars()];
    let mut n_free = 0usize;
    for (g, slot) in assignment.iter().enumerate() {
        if slot.is_none() {
            g2l[g] = n_free;
            n_free += 1;
        }
    }

    if n_free == 0 {
        return instance.offset.to_f64();
    }

    let mut base_offset = instance.offset.to_f64();
    let mut active_terms: Vec<(f64, Vec<usize>)> = Vec::new();

    for term in &instance.terms {
        let mut coeff = term.coeff.to_f64();
        let mut locals = Vec::<usize>::new();
        let mut zero = false;

        for &idx in &term.indices {
            if let Some(v) = assignment[idx] {
                let vf = v.to_f64();
                match var_type {
                    VarType::Bin => {
                        if vf == 0.0 {
                            zero = true;
                            break;
                        }
                    }
                    VarType::Spin => coeff *= vf,
                }
            } else {
                locals.push(g2l[idx]);
            }
        }
        if zero {
            continue;
        }
        locals.sort_unstable();
        if matches!(var_type, VarType::Bin) {
            locals.dedup();
        }
        if locals.is_empty() {
            base_offset += coeff;
        } else {
            active_terms.push((coeff, locals));
        }
    }

    if active_terms.is_empty() {
        return base_offset;
    }

    // Bypass max_vars at the root (all variables still free): the LP is most
    // valuable there and we only pay this cost once in the BnB.
    let is_root = n_free == assignment.len();
    if !is_root && n_free > cfg.max_vars {
        let trivial: f64 = active_terms
            .iter()
            .map(|(c, _)| match var_type {
                VarType::Bin => c.min(0.0),
                VarType::Spin => -c.abs(),
            })
            .sum();
        return base_offset + trivial;
    }

    match solve_sa_lp(n_free, var_type, &active_terms, cfg) {
        Ok(lp_val) => {
            log::debug!(
                "SA LP bound: {:.6} (base {:.6})",
                base_offset + lp_val,
                base_offset
            );
            base_offset + lp_val
        }
        Err(e) => {
            log::debug!("SA LP failed ({e}), falling back to trivial");
            let trivial: f64 = active_terms
                .iter()
                .map(|(c, _)| match var_type {
                    VarType::Bin => c.min(0.0),
                    VarType::Spin => -c.abs(),
                })
                .sum();
            base_offset + trivial
        }
    }
}

/// Wrapper called from `bounds/mod.rs`.
pub(crate) fn compute<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    node: &Node<C>,
    cfg: &SheraliAdamsConfig,
) -> C {
    let ov = node.to_option_vec(instance);
    C::from_f64_lb(sherali_adams_lower_bound(instance, &ov, cfg))
}

// ── chain construction ────────────────────────────────────────────────────────

/// Build the set of product monomials and chain steps for the given level.
///
/// Returns `(monomials_in_creation_order, chain_steps)`.
///
/// **Invariant**: monomials are ordered by degree (ascending) so that when
/// we create a degree-k product its degree-(k−1) sub-products already exist.
fn build_chain(
    level: usize,
    active_terms: &[(f64, Vec<usize>)],
) -> (Vec<Vec<usize>>, Vec<(Vec<usize>, ChainFactor, usize)>) {
    let mut mono_set: BTreeSet<Vec<usize>> = BTreeSet::new();

    // ── collect monomials ─────────────────────────────────────────────────────

    for (_, vars) in active_terms {
        if vars.len() < 2 {
            continue;
        }
        // Always add the left-to-right chain (needed for the objective variable).
        for k in 2..=vars.len() {
            mono_set.insert(vars[..k].to_vec());
        }

        if level >= 1 {
            // All C(k,2) pairs within this term.
            for i in 0..vars.len() {
                for j in i + 1..vars.len() {
                    mono_set.insert(vec![vars[i], vars[j]]);
                }
            }
        }

        if level >= 2 {
            // All C(k,3) triples within this term.
            for i in 0..vars.len() {
                for j in i + 1..vars.len() {
                    for k in j + 1..vars.len() {
                        mono_set.insert(vec![vars[i], vars[j], vars[k]]);
                    }
                }
            }
        }
    }

    // Cross-term enrichment for level ≥ 2: for every pair of terms sharing ≥1
    // variable, add all pairs from their union.
    if level >= 2 {
        let term_vars: Vec<&[usize]> = active_terms.iter().map(|(_, v)| v.as_slice()).collect();
        for i in 0..term_vars.len() {
            for j in i + 1..term_vars.len() {
                if !terms_share_variable(term_vars[i], term_vars[j]) {
                    continue;
                }
                let union = sorted_union(term_vars[i], term_vars[j]);
                for p in 0..union.len() {
                    for q in p + 1..union.len() {
                        mono_set.insert(vec![union[p], union[q]]);
                    }
                }
            }
        }
    }

    // ── sort by degree (ascending) ────────────────────────────────────────────
    let mut monomials: Vec<Vec<usize>> = mono_set.into_iter().collect();
    monomials.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

    // ── build chain steps ─────────────────────────────────────────────────────
    // Build a set of all monomials for quick lookup.
    let mono_set_ref: std::collections::HashSet<&[usize]> =
        monomials.iter().map(|m| m.as_slice()).collect();

    let mut chain_steps: Vec<(Vec<usize>, ChainFactor, usize)> = Vec::new();

    for m in &monomials {
        if m.len() == 2 {
            // Single canonical step: w[a,b] = x[a] · x[b].
            chain_steps.push((m.clone(), ChainFactor::X(m[0]), m[1]));
        } else {
            // For each element i_j ∈ m: step w[m] = w[m\{i_j}] · x[i_j].
            // Only add the step if m\{i_j} is available in the monomial set.
            for k in 0..m.len() {
                let mut sub = m.clone();
                let removed = sub.remove(k); // sub = m \ {m[k]}
                if mono_set_ref.contains(sub.as_slice()) {
                    let left = if sub.len() == 1 {
                        ChainFactor::X(sub[0])
                    } else {
                        ChainFactor::Prod(sub)
                    };
                    chain_steps.push((m.clone(), left, removed));
                }
            }
        }
    }

    (monomials, chain_steps)
}

fn terms_share_variable(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn sorted_union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

// ── separation oracle (pure Rust, no SCIP) ───────────────────────────────────

fn sa_violations_for_step(
    var_type: VarType,
    w: f64,
    a: f64,
    b: f64,
    prefix: &[usize],
    left: &ChainFactor,
    right_x: usize,
    tol: f64,
) -> Vec<SaViolation> {
    let mut out = Vec::new();

    let push = |out: &mut Vec<SaViolation>, cut: SaCutKind, v: f64| {
        if v > tol {
            out.push(SaViolation {
                prefix: prefix.to_vec(),
                left: left.clone(),
                right_x,
                cut,
                violation: v,
            });
        }
    };

    match var_type {
        VarType::Bin => {
            push(&mut out, SaCutKind::WLeqA, w - a);
            push(&mut out, SaCutKind::WLeqB, w - b);
            push(&mut out, SaCutKind::WGeqAplusB, a + b - 1.0 - w);
        }
        VarType::Spin => {
            push(&mut out, SaCutKind::WLeqA, w - a + b - 1.0);
            push(&mut out, SaCutKind::WLeqB, w + a - b - 1.0);
            push(&mut out, SaCutKind::WGeqAplusB, a + b - 1.0 - w);
            push(&mut out, SaCutKind::WGeqMinusAMinusB, -a - b - 1.0 - w);
        }
    }

    out
}

// ── SCIP LP solver with separation loop ──────────────────────────────────────

fn scip_call(code: ffi::SCIP_Retcode, op: &str) -> Result<(), String> {
    if code == ffi::SCIP_Retcode_SCIP_OKAY {
        Ok(())
    } else {
        Err(format!("{op} failed with retcode {code}"))
    }
}

fn solve_sa_lp(
    m: usize,
    var_type: VarType,
    active_terms: &[(f64, Vec<usize>)],
    cfg: &SheraliAdamsConfig,
) -> Result<f64, String> {
    unsafe {
        let mut scip: *mut ffi::SCIP = ptr::null_mut();
        scip_call(ffi::SCIPcreate(&mut scip), "SCIPcreate")?;

        let result: Result<f64, String> = (|| {
            scip_call(ffi::SCIPincludeDefaultPlugins(scip), "SCIPincludeDefaultPlugins")?;
            let prob = CString::new("sa_lp").map_err(|e| e.to_string())?;
            scip_call(ffi::SCIPcreateProbBasic(scip, prob.as_ptr()), "SCIPcreateProbBasic")?;
            scip_call(
                ffi::SCIPsetObjsense(scip, ffi::SCIP_Objsense_SCIP_OBJSENSE_MINIMIZE),
                "SCIPsetObjsense",
            )?;
            let vparam = CString::new("display/verblevel").map_err(|e| e.to_string())?;
            let _ = ffi::SCIPsetIntParam(scip, vparam.as_ptr(), 0);

            let (xlb, xub) = match var_type {
                VarType::Bin => (0.0_f64, 1.0_f64),
                VarType::Spin => (-1.0_f64, 1.0_f64),
            };

            // ── x[i] variables ────────────────────────────────────────────────
            let mut x_vars: Vec<*mut ffi::SCIP_VAR> = Vec::with_capacity(m);
            for i in 0..m {
                let name = CString::new(format!("x{i}")).map_err(|e| e.to_string())?;
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateVarBasic(scip, &mut v, name.as_ptr(), xlb, xub, 0.0,
                        ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS),
                    "SCIPcreateVarBasic(x)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(x)")?;
                x_vars.push(v);
            }

            // ── product variables ─────────────────────────────────────────────
            let (monomials, chain_steps) = build_chain(cfg.level, active_terms);

            let mut prod_vars: HashMap<Vec<usize>, *mut ffi::SCIP_VAR> = HashMap::new();
            for mono in &monomials {
                let tag = mono.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_");
                let name = CString::new(format!("w{tag}")).map_err(|e| e.to_string())?;
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateVarBasic(scip, &mut v, name.as_ptr(), xlb, xub, 0.0,
                        ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS),
                    "SCIPcreateVarBasic(w)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(w)")?;
                prod_vars.insert(mono.clone(), v);
            }

            // ── objective coefficients ─────────────────────────────────────────
            for (coeff, vars) in active_terms {
                let obj_var = if vars.len() == 1 {
                    x_vars[vars[0]]
                } else {
                    *prod_vars.get(vars.as_slice())
                        .ok_or_else(|| format!("missing product var for {vars:?}"))?
                };
                let cur = ffi::SCIPvarGetObj(obj_var);
                scip_call(ffi::SCIPchgVarObj(scip, obj_var, cur + coeff), "SCIPchgVarObj")?;
            }

            // ── register all McCormick constraints as LAZY cuts ────────────────
            //
            // Using SCIPcreateConsLinear with initial=FALSE / dynamic=TRUE /
            // removable=TRUE means SCIP treats these as lazy constraints:
            //   • They are NOT added to the initial LP (initial=FALSE)
            //   • SCIP's built-in linear constraint separator checks them each LP
            //     solve and adds violated ones (separate=TRUE)
            //   • After adding a violated cut SCIP re-solves the LP from the
            //     current basis (warm-started) — no SCIPfreeTransform restart
            //   • Non-binding cuts can be dropped from the LP (removable=TRUE)
            //
            // This replaces the old manual SCIPfreeTransform + SCIPsolve loop.
            let mut cons_idx = 0usize;
            for (prefix, left, right_x) in &chain_steps {
                let w_var = prod_vars[prefix];
                let a_var: *mut ffi::SCIP_VAR = match left {
                    ChainFactor::X(i) => x_vars[*i],
                    ChainFactor::Prod(p) => prod_vars[p],
                };
                let b_var = x_vars[*right_x];
                add_lazy_mccormick(scip, var_type, w_var, a_var, b_var, &mut cons_idx)?;
            }

            // Map cfg.max_rounds to SCIP's root separation round limit.
            let sep_root = CString::new("separating/maxroundsroot").map_err(|e| e.to_string())?;
            let _ = ffi::SCIPsetIntParam(scip, sep_root.as_ptr(), cfg.max_rounds as i32);

            let tlim_param = CString::new("limits/time").map_err(|e| e.to_string())?;
            let tlim = (0.1 + 0.005 * (m + monomials.len()) as f64).clamp(0.5, 30.0);
            let _ = ffi::SCIPsetRealParam(scip, tlim_param.as_ptr(), tlim);

            // Single solve — SCIP handles cut separation and LP warm-restart internally.
            scip_call(ffi::SCIPsolve(scip), "SCIPsolve")?;

            let dual = ffi::SCIPgetDualbound(scip);
            let trivial: f64 = active_terms
                .iter()
                .map(|(c, _)| match var_type {
                    VarType::Bin => c.min(0.0),
                    VarType::Spin => -c.abs(),
                })
                .sum();
            let lp_val = if dual.is_finite() { dual } else { trivial };

            for v in &mut x_vars {
                let _ = ffi::SCIPreleaseVar(scip, v);
            }
            let mut prod_var_list: Vec<*mut ffi::SCIP_VAR> = prod_vars.into_values().collect();
            for v in &mut prod_var_list {
                let _ = ffi::SCIPreleaseVar(scip, v);
            }

            Ok(lp_val)
        })();

        let free_res = scip_call(ffi::SCIPfree(&mut scip), "SCIPfree");
        if let Err(e) = free_res {
            log::debug!("SCIP free failed: {e}");
        }
        result
    }
}

/// Add all McCormick inequalities for step w = a·b as **lazy** constraints.
///
/// `initial=FALSE, separate=TRUE, dynamic=TRUE, removable=TRUE` tells SCIP to
/// treat these as lazy cuts: not in the initial LP, added by the built-in
/// linear constraint separator when violated, resolved from the current basis.
unsafe fn add_lazy_mccormick(
    scip: *mut ffi::SCIP,
    var_type: VarType,
    w: *mut ffi::SCIP_VAR,
    a: *mut ffi::SCIP_VAR,
    b: *mut ffi::SCIP_VAR,
    idx: &mut usize,
) -> Result<(), String> {
    match var_type {
        VarType::Bin => {
            // w ≤ a
            unsafe { add_lazy_le(scip, &mut [w, a], &mut [1.0, -1.0], 0.0, *idx)? };
            *idx += 1;
            // w ≤ b
            unsafe { add_lazy_le(scip, &mut [w, b], &mut [1.0, -1.0], 0.0, *idx)? };
            *idx += 1;
            // w ≥ a + b − 1
            unsafe { add_lazy_le(scip, &mut [w, a, b], &mut [-1.0, 1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
        }
        VarType::Spin => {
            // w ≤ 1 + a − b
            unsafe { add_lazy_le(scip, &mut [w, a, b], &mut [1.0, -1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
            // w ≤ 1 − a + b
            unsafe { add_lazy_le(scip, &mut [w, a, b], &mut [1.0, 1.0, -1.0], 1.0, *idx)? };
            *idx += 1;
            // w ≥ a + b − 1
            unsafe { add_lazy_le(scip, &mut [w, a, b], &mut [-1.0, 1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
            // w ≥ −a − b − 1
            unsafe { add_lazy_le(scip, &mut [w, a, b], &mut [-1.0, -1.0, -1.0], 1.0, *idx)? };
            *idx += 1;
        }
    }
    Ok(())
}

/// Create a single lazy inequality: Σ coefs[i]·vars[i] ≤ rhs.
///
/// Flags:  initial=0  separate=1  enforce=1  check=1  propagate=1
///         local=0  modifiable=0  dynamic=1  removable=1  stickingatnode=0
unsafe fn add_lazy_le(
    scip: *mut ffi::SCIP,
    vars: &mut [*mut ffi::SCIP_VAR],
    coefs: &mut [f64],
    rhs: f64,
    idx: usize,
) -> Result<(), String> {
    let name = CString::new(format!("mc{idx}")).map_err(|e| e.to_string())?;
    let mut cons: *mut ffi::SCIP_CONS = ptr::null_mut();
    scip_call(
        unsafe {
            ffi::SCIPcreateConsLinear(
                scip, &mut cons, name.as_ptr(),
                vars.len() as i32, vars.as_mut_ptr(), coefs.as_mut_ptr(),
                f64::NEG_INFINITY, rhs,
                0, // initial   = FALSE  → not in initial LP
                1, // separate  = TRUE   → separated lazily when violated
                1, // enforce   = TRUE
                1, // check     = TRUE
                1, // propagate = TRUE
                0, // local     = FALSE
                0, // modifiable= FALSE
                1, // dynamic   = TRUE   → treated as a removable lazy cut
                1, // removable = TRUE   → SCIP drops it from LP when non-binding
                0, // stickingatnode = FALSE
            )
        },
        "SCIPcreateConsLinear",
    )?;
    scip_call(unsafe { ffi::SCIPaddCons(scip, cons) }, "SCIPaddCons")?;
    let mut crel = cons;
    scip_call(unsafe { ffi::SCIPreleaseCons(scip, &mut crel) }, "SCIPreleaseCons")?;
    Ok(())
}


// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::Bin, instance::HuboInstance, term::Term};

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

    fn none_assignment(n: usize) -> Vec<Option<f64>> {
        vec![None; n]
    }

    #[test]
    fn build_chain_level0_only_lr_chain() {
        // For term [0,1,2] at level 0: should have w[0,1] and w[0,1,2] only.
        let terms = vec![(1.0_f64, vec![0usize, 1, 2])];
        let (monos, steps) = build_chain(0, &terms);
        assert!(monos.contains(&vec![0, 1]));
        assert!(monos.contains(&vec![0, 1, 2]));
        // Should NOT have w[0,2] or w[1,2] at level 0.
        assert!(
            !monos.contains(&vec![0, 2]),
            "level-0 should not add w[0,2]"
        );
        assert!(
            !monos.contains(&vec![1, 2]),
            "level-0 should not add w[1,2]"
        );
        // Steps: (w[0,1], X(0), 1) and (w[0,1,2], Prod([0,1]), 2).
        assert_eq!(
            steps.len(),
            2,
            "level-0 should have exactly 2 steps for term [0,1,2]"
        );
        let _ = steps; // suppress unused warning
    }

    #[test]
    fn build_chain_level1_all_pairs() {
        // For term [0,1,2] at level 1: should have all three pairs.
        let terms = vec![(1.0_f64, vec![0usize, 1, 2])];
        let (monos, steps) = build_chain(1, &terms);
        assert!(monos.contains(&vec![0, 1]));
        assert!(monos.contains(&vec![0, 2]));
        assert!(monos.contains(&vec![1, 2]));
        assert!(monos.contains(&vec![0, 1, 2]));
        // w[0,1,2] can now be decomposed in all 3 orderings since all pairs exist.
        let triple_steps: Vec<_> = steps
            .iter()
            .filter(|(p, _, _)| *p == vec![0, 1, 2])
            .collect();
        assert_eq!(
            triple_steps.len(),
            3,
            "level-1 should have 3 orderings for the triple"
        );
    }

    #[test]
    fn build_chain_level2_cross_term_pairs() {
        // Terms [0,1] and [1,2] share variable 1: level-2 should add w[0,2].
        let terms = vec![(1.0_f64, vec![0usize, 1]), (1.0_f64, vec![1usize, 2])];
        let (monos, _steps) = build_chain(2, &terms);
        assert!(monos.contains(&vec![0, 1]));
        assert!(monos.contains(&vec![1, 2]));
        assert!(
            monos.contains(&vec![0, 2]),
            "cross-term pair [0,2] must be added at level 2"
        );
    }

    #[test]
    fn separation_oracle_finds_violated_mccormick() {
        let x_vals = vec![0.5, 0.5];
        let mut prod_vals = HashMap::new();
        prod_vals.insert(vec![0usize, 1], 0.8_f64);
        let chain = vec![(vec![0usize, 1], ChainFactor::X(0), 1usize)];
        let viols = separate_sa_cuts(&x_vals, &prod_vals, &chain, VarType::Bin, 1e-9);

        assert!(!viols.is_empty(), "should find violations");
        assert!(
            viols.iter().any(|v| v.cut == SaCutKind::WLeqA),
            "WLeqA must fire"
        );
        assert!(
            viols.iter().any(|v| v.cut == SaCutKind::WLeqB),
            "WLeqB must fire"
        );
        assert!(
            !viols.iter().any(|v| v.cut == SaCutKind::WGeqAplusB),
            "WGeqAplusB must not fire"
        );
        assert!(
            (viols[0].violation - 0.3).abs() < 1e-9,
            "max violation should be 0.3"
        );
    }

    #[test]
    fn sa_bound_never_exceeds_optimum() {
        let inst = make_bin(2, vec![(vec![0, 1], -1.0)]);
        let lb =
            sherali_adams_lower_bound(&inst, &none_assignment(2), &SheraliAdamsConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ −1");
    }

    #[test]
    fn sa_level1_tighter_than_level0_on_degree3_term() {
        // min −x0·x1·x2: level-1 adds w[0,2] and w[1,2] and all 3 orderings,
        // which should give a bound at least as tight as level-0.
        let inst = make_bin(3, vec![(vec![0, 1, 2], -1.0)]);
        let cfg0 = SheraliAdamsConfig {
            level: 0,
            max_rounds: 10,
            ..Default::default()
        };
        let cfg1 = SheraliAdamsConfig {
            level: 1,
            max_rounds: 10,
            ..Default::default()
        };
        let lb0 = sherali_adams_lower_bound(&inst, &none_assignment(3), &cfg0);
        let lb1 = sherali_adams_lower_bound(&inst, &none_assignment(3), &cfg1);
        assert!(
            lb0 <= lb1 + 1e-9,
            "level-1 ({lb1}) must be ≥ level-0 ({lb0})"
        );
        assert!(lb1 <= -1.0 + 1e-6, "lb1={lb1} must be ≤ −1");
    }

    #[test]
    fn sa_bound_with_fixed_variable() {
        let inst = make_bin(3, vec![(vec![0, 1, 2], -1.0)]);
        let assignment = vec![None, None, Some(1.0_f64)];
        let lb = sherali_adams_lower_bound(&inst, &assignment, &SheraliAdamsConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ −1 with x2 fixed to 1");
    }
}
