//! RLT-1 (Reformulation–Linearization Technique) LP lower bound for HUBO.
//!
//! Builds a linear programme (LP) by introducing one product variable w_S per
//! monomial S appearing in the instance, linearising the objective, and adding
//! McCormick inequalities that enforce w_S = Π_{i∈S} x_i at every LP vertex:
//!
//! Binary (x_i ∈ [0,1]):  w ≤ a,  w ≤ b,  w ≥ a+b−1,  w ≥ 0
//! Spin   (x_i ∈ [−1,1]): w ≤ 1+a−b,  w ≤ 1−a+b,  w ≥ a+b−1,  w ≥ −a−b−1
//!
//! Higher-degree terms (degree k) are handled by a left-to-right product chain:
//!   w_{i₁i₂} = x_{i₁}·x_{i₂},  w_{i₁i₂i₃} = w_{i₁i₂}·x_{i₃}, …
//!
//! The LP dual objective equals the Lagrangian relaxation dual optimum (strong
//! duality holds for LPs), giving a valid lower bound on the HUBO problem that
//! is tighter than the naive per-term minimum bound.

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use russcip::ffi;

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;
use crate::solver::bnb::Node;

/// Configuration for the RLT-1 LP lower bound.
#[derive(Debug, Clone)]
pub struct RltConfig {
    /// Skip computation if free variables exceed this (LP grows large).
    pub max_vars: usize,
}

impl Default for RltConfig {
    fn default() -> Self {
        Self { max_vars: 200 }
    }
}

/// Newtype used as the `LowerBound` impl.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct RltLp(pub RltConfig);


/// Compute the RLT-1 LP lower bound for the given (possibly partial) assignment.
pub fn rlt_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
    cfg: &RltConfig,
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

    if n_free > cfg.max_vars {
        let trivial: f64 = active_terms
            .iter()
            .map(|(c, _)| match var_type {
                VarType::Bin => c.min(0.0),
                VarType::Spin => -c.abs(),
            })
            .sum();
        return base_offset + trivial;
    }

    match solve_rlt_lp(n_free, var_type, &active_terms) {
        Ok(lp_val) => {
            log::debug!("RLT LP bound: {:.6} (base {:.6})", base_offset + lp_val, base_offset);
            base_offset + lp_val
        }
        Err(e) => {
            log::debug!("RLT LP failed ({e}), falling back to trivial");
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
    cfg: &RltConfig,
) -> C {
    let ov = node.to_option_vec(instance);
    C::from_f64_lb(rlt_lower_bound(instance, &ov, cfg))
}

// ── internal LP solver ────────────────────────────────────────────────────────

fn scip_call(code: ffi::SCIP_Retcode, op: &str) -> Result<(), String> {
    if code == ffi::SCIP_Retcode_SCIP_OKAY {
        Ok(())
    } else {
        Err(format!("{op} failed with retcode {code}"))
    }
}

/// Build and solve the RLT-1 LP.  Returns the LP optimal value (without base offset).
fn solve_rlt_lp(
    m: usize,
    var_type: VarType,
    active_terms: &[(f64, Vec<usize>)],
) -> Result<f64, String> {
    unsafe {
        let mut scip: *mut ffi::SCIP = ptr::null_mut();
        scip_call(ffi::SCIPcreate(&mut scip), "SCIPcreate")?;

        let result: Result<f64, String> = (|| {
            scip_call(
                ffi::SCIPincludeDefaultPlugins(scip),
                "SCIPincludeDefaultPlugins",
            )?;

            let prob = CString::new("rlt_lp").map_err(|e| e.to_string())?;
            scip_call(
                ffi::SCIPcreateProbBasic(scip, prob.as_ptr()),
                "SCIPcreateProbBasic",
            )?;
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

            // ── x variables ───────────────────────────────────────────────────
            let mut x_vars: Vec<*mut ffi::SCIP_VAR> = Vec::with_capacity(m);
            for i in 0..m {
                let name = CString::new(format!("x{i}")).map_err(|e| e.to_string())?;
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateVarBasic(
                        scip,
                        &mut v,
                        name.as_ptr(),
                        xlb,
                        xub,
                        0.0,
                        ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                    ),
                    "SCIPcreateVarBasic(x)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(x)")?;
                x_vars.push(v);
            }

            // ── product variables and McCormick constraints ────────────────────
            // For each term of degree k ≥ 2, build the left-to-right chain:
            //   prefix [i0,i1] → w_{i0,i1}
            //   prefix [i0,i1,i2] → w_{i0,i1,i2} = w_{i0,i1} · x_{i2}
            //   …
            // Deduplicate shared prefixes across terms.
            let mut prod_vars: HashMap<Vec<usize>, *mut ffi::SCIP_VAR> = HashMap::new();
            // (w_var, left_factor_var, right_x_var) — for McCormick constraint creation.
            let mut mccormick_triples: Vec<(*mut ffi::SCIP_VAR, *mut ffi::SCIP_VAR, *mut ffi::SCIP_VAR)> =
                Vec::new();

            for (_, vars) in active_terms {
                if vars.len() < 2 {
                    continue;
                }
                for k in 2..=vars.len() {
                    let prefix = vars[..k].to_vec();
                    if prod_vars.contains_key(&prefix) {
                        continue;
                    }

                    let tag = prefix
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join("_");
                    let name = CString::new(format!("w{tag}")).map_err(|e| e.to_string())?;
                    let mut w: *mut ffi::SCIP_VAR = ptr::null_mut();
                    scip_call(
                        ffi::SCIPcreateVarBasic(
                            scip,
                            &mut w,
                            name.as_ptr(),
                            xlb,
                            xub,
                            0.0,
                            ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                        ),
                        "SCIPcreateVarBasic(w)",
                    )?;
                    scip_call(ffi::SCIPaddVar(scip, w), "SCIPaddVar(w)")?;

                    let a: *mut ffi::SCIP_VAR = if k == 2 {
                        x_vars[prefix[0]]
                    } else {
                        *prod_vars
                            .get(&prefix[..k - 1])
                            .ok_or_else(|| format!("missing prefix {:?}", &prefix[..k - 1]))?
                    };
                    let b: *mut ffi::SCIP_VAR = x_vars[prefix[k - 1]];

                    mccormick_triples.push((w, a, b));
                    prod_vars.insert(prefix, w);
                }
            }

            // Add McCormick constraints.
            let mut cons_idx = 0usize;
            for (w, a, b) in &mccormick_triples {
                add_mccormick(scip, var_type, *w, *a, *b, &mut cons_idx)?;
            }

            // ── objective coefficients ─────────────────────────────────────────
            for (coeff, vars) in active_terms {
                let obj_var = if vars.len() == 1 {
                    x_vars[vars[0]]
                } else {
                    *prod_vars
                        .get(vars.as_slice())
                        .ok_or_else(|| format!("missing product var for {vars:?}"))?
                };
                let cur = ffi::SCIPvarGetObj(obj_var);
                scip_call(
                    ffi::SCIPchgVarObj(scip, obj_var, cur + coeff),
                    "SCIPchgVarObj",
                )?;
            }

            // ── solve ──────────────────────────────────────────────────────────
            let tlim_param = CString::new("limits/time").map_err(|e| e.to_string())?;
            let n_prod = prod_vars.len();
            let tlim = (0.1 + 0.005 * (m + n_prod) as f64).clamp(0.5, 10.0);
            let _ = ffi::SCIPsetRealParam(scip, tlim_param.as_ptr(), tlim);

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

            // ── release variables ──────────────────────────────────────────────
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

/// Add the four McCormick inequalities for w = a · b.
///
/// Binary: w ≤ a,  w ≤ b,  w ≥ a+b−1  (w ≥ 0 from var bounds)
/// Spin:   w ≤ 1+a−b,  w ≤ 1−a+b,  w ≥ a+b−1,  w ≥ −a−b−1
unsafe fn add_mccormick(
    scip: *mut ffi::SCIP,
    var_type: VarType,
    w: *mut ffi::SCIP_VAR,
    a: *mut ffi::SCIP_VAR,
    b: *mut ffi::SCIP_VAR,
    idx: &mut usize,
) -> Result<(), String> {
    match var_type {
        VarType::Bin => {
            // w − a ≤ 0
            unsafe { add_lin_le(scip, &mut [w, a], &mut [1.0, -1.0], 0.0, *idx)? };
            *idx += 1;
            // w − b ≤ 0
            unsafe { add_lin_le(scip, &mut [w, b], &mut [1.0, -1.0], 0.0, *idx)? };
            *idx += 1;
            // −w + a + b ≤ 1
            unsafe { add_lin_le(scip, &mut [w, a, b], &mut [-1.0, 1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
        }
        VarType::Spin => {
            // w − a + b ≤ 1  (w ≤ 1+a−b)
            unsafe { add_lin_le(scip, &mut [w, a, b], &mut [1.0, -1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
            // w + a − b ≤ 1  (w ≤ 1−a+b)
            unsafe { add_lin_le(scip, &mut [w, a, b], &mut [1.0, 1.0, -1.0], 1.0, *idx)? };
            *idx += 1;
            // −w + a + b ≤ 1  (w ≥ a+b−1)
            unsafe { add_lin_le(scip, &mut [w, a, b], &mut [-1.0, 1.0, 1.0], 1.0, *idx)? };
            *idx += 1;
            // −w − a − b ≤ 1  (w ≥ −a−b−1)
            unsafe { add_lin_le(scip, &mut [w, a, b], &mut [-1.0, -1.0, -1.0], 1.0, *idx)? };
            *idx += 1;
        }
    }
    Ok(())
}

/// Add a single linear inequality: Σ coefs[i]·vars[i] ≤ rhs.
unsafe fn add_lin_le(
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
            ffi::SCIPcreateConsBasicLinear(
                scip,
                &mut cons,
                name.as_ptr(),
                vars.len() as i32,
                vars.as_mut_ptr(),
                coefs.as_mut_ptr(),
                f64::NEG_INFINITY,
                rhs,
            )
        },
        "SCIPcreateConsBasicLinear",
    )?;
    scip_call(unsafe { ffi::SCIPaddCons(scip, cons) }, "SCIPaddCons")?;
    let mut crel = cons;
    scip_call(
        unsafe { ffi::SCIPreleaseCons(scip, &mut crel) },
        "SCIPreleaseCons",
    )?;
    Ok(())
}

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

    fn assignment_none(n: usize) -> Vec<Option<f64>> {
        vec![None; n]
    }

    #[test]
    fn rlt_bound_never_exceeds_optimum_small_bin() {
        // min -x0·x1  s.t. x0,x1 ∈ {0,1}  → optimum = -1
        let inst = make_bin(2, vec![(vec![0, 1], -1.0)]);
        let lb = rlt_lower_bound(&inst, &assignment_none(2), &RltConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ -1");
    }

    #[test]
    fn rlt_bound_is_tighter_than_trivial_for_conflicting_pair() {
        // min x0·x1 − x0 − x1  → trivial = -1-1 = -2; RLT-LP optimum = -1
        // (set x0=x1=1 → value 1-1-1=-1; RLT LP: w≤x0,w≤x1,w≥x0+x1-1 + obj = w-x0-x1)
        let inst = make_bin(
            2,
            vec![(vec![0, 1], 1.0), (vec![0], -1.0), (vec![1], -1.0)],
        );
        let trivial = -2.0_f64;
        let lb = rlt_lower_bound(&inst, &assignment_none(2), &RltConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ -1 (optimum)");
        assert!(lb >= trivial - 1e-6, "lb={lb} should be ≥ trivial {trivial}");
        // RLT LP should give −1, which is tighter than −2
        assert!(lb > trivial + 0.5, "RLT LP ({lb}) should be tighter than trivial ({trivial})");
    }

    #[test]
    fn rlt_bound_handles_fixed_variables() {
        // min x0·x1·x2; fix x2=1 → min x0·x1 → same test as above with offset 0
        let inst = make_bin(3, vec![(vec![0, 1, 2], -1.0)]);
        let assignment = vec![None, None, Some(1.0_f64)];
        let lb = rlt_lower_bound(&inst, &assignment, &RltConfig::default());
        assert!(lb <= -1.0 + 1e-6, "lb={lb} must be ≤ -1");
    }
}
