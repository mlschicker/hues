//! SCIP-backed Lasserre hierarchy lower bounds for HUBO instances.
//!
//! This module builds an order-d moment relaxation using an instance-specific
//! monomial basis, then solves it with SCIP as a nonlinear program:
//!
//! - moment variables y_S (bounded by variable domain)
//! - factorization constraints M_d(y) = L L^T to enforce PSD
//!
//! It also exposes an incremental state that can fix a single variable,
//! rebuild the reduced relaxation, and warm-start the next solve.

use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::ptr;

use russcip::ffi;

use crate::coeff::Coeff;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

/// Configuration for the Lasserre / moment-SDP hierarchy.
///
/// The hierarchy is parameterised by `level` (Lasserre order `d`):
///
/// | level | basis degree | captures terms up to degree | basis size (typical) |
/// |-------|-------------|----------------------------|----------------------|
/// |   1   |      1      |           2                |   1 + n              |
/// |   2   |      2      |           4                |   1 + n + pairs      |
/// |   3   |      3      |           6                |   …                  |
///
/// "Possible" monomials: only sub-monomials of instance terms (and their
/// pairwise overlapping unions) are added to the basis, keeping the matrix
/// sparse relative to the full $\binom{n}{d}$ Lasserre basis.
#[derive(Debug, Clone)]
pub struct LasserreConfig {
    /// Hierarchy level (Lasserre order d ≥ 1).
    /// Level 1 = standard LP moment relaxation; level 2 adds pairwise and
    /// cross-term correlations; higher levels further tighten the bound.
    pub level: usize,
    /// Legacy field kept for CLI compatibility; unused in SCIP formulation.
    pub max_iter: usize,
    /// Skip SDP solve when number of free variables exceeds this threshold.
    pub max_vars: usize,
    /// Maximum number of basis elements (= moment matrix dimension).
    /// Caps both direct seeding and cross-term enrichment.  The resulting
    /// moment matrix is at most `max_basis × max_basis` with
    /// `max_basis*(max_basis+1)/2` PSD equality constraints for SCIP.
    /// Tuning: for 40-variable instances at level 2, ~80 gives a tractable
    /// SDP; higher values improve the bound but increase solve time.
    pub max_basis: usize,
    /// Legacy field kept for CLI compatibility; unused in SCIP formulation.
    pub step_size: f64,
}

impl Default for LasserreConfig {
    fn default() -> Self {
        Self {
            level: 2,
            max_iter: 100,
            max_vars: 50,
            max_basis: 80,
            step_size: 0.1,
        }
    }
}

/// Configuration for the exact (dense) Lasserre relaxation.
#[derive(Debug, Clone)]
pub struct ExactLasserreConfig {
    /// Hierarchy level (see [`LasserreConfig::level`]).
    pub level: usize,
    pub max_vars: usize,
    /// Maximum basis size (see [`LasserreConfig::max_basis`]).
    pub max_basis: usize,
}

impl Default for ExactLasserreConfig {
    fn default() -> Self {
        Self {
            level: 2,
            max_vars: 50,
            max_basis: 80,
        }
    }
}

/// Stateful Lasserre relaxation object that supports incremental updates.
#[derive(Debug, Clone)]
pub struct LasserreSdpState {
    var_type: VarType,
    d: usize,
    max_basis: usize,
    base_offset: f64,
    free_globals: Vec<usize>,
    // Active terms over free local indices (sorted, deduplicated for binary)
    active_terms: Vec<(f64, Vec<usize>)>,
    // Warmstart cache from previous solve.
    warm_moments: HashMap<Vec<usize>, f64>,
    warm_l: Vec<f64>,
}

pub fn lasserre_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
    cfg: &LasserreConfig,
) -> f64 {
    let mut state = LasserreSdpState::new(
        instance,
        assignment,
        cfg.level.max(2),
        cfg.max_vars,
        cfg.max_basis,
    );
    state.solve()
}

pub fn lasserre_exact_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
    cfg: &ExactLasserreConfig,
) -> f64 {
    let mut state = LasserreSdpState::new(
        instance,
        assignment,
        cfg.level.max(1),
        cfg.max_vars,
        cfg.max_basis,
    );
    state.solve()
}

impl LasserreSdpState {
    pub fn new<C: Coeff, V: VarDomain>(
        instance: &HuboInstance<C, V>,
        assignment: &[Option<C>],
        d: usize,
        _max_vars: usize,
        max_basis: usize,
    ) -> Self {
        let var_type = V::VAR_TYPE;

        let mut free_globals = Vec::new();
        let mut g2l = vec![usize::MAX; instance.n_vars()];
        for (g, slot) in assignment.iter().enumerate() {
            if slot.is_none() {
                g2l[g] = free_globals.len();
                free_globals.push(g);
            }
        }

        let mut base_offset = instance.offset.to_f64();
        let mut active_terms = Vec::<(f64, Vec<usize>)>::new();

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

        Self {
            var_type,
            d,
            max_basis,
            base_offset,
            free_globals,
            active_terms,
            warm_moments: HashMap::new(),
            warm_l: Vec::new(),
        }
    }

    /// Fix one free variable and resolve with warmstart.
    pub fn update_with_fixed_variable<C: Coeff>(&mut self, global_var: usize, value: C) -> f64 {
        let Some(local_idx) = self.free_globals.iter().position(|&g| g == global_var) else {
            return self.solve();
        };
        self.free_globals.remove(local_idx);

        let mut new_terms = Vec::<(f64, Vec<usize>)>::new();
        let mut base_add = 0.0;
        let val = value.to_f64();

        for (coeff, vars) in &self.active_terms {
            if let Some(pos) = vars.iter().position(|&v| v == local_idx) {
                let mut c = *coeff;
                match self.var_type {
                    VarType::Bin => {
                        if val == 0.0 {
                            continue;
                        }
                    }
                    VarType::Spin => c *= val,
                }

                let mut nv = vars.clone();
                nv.remove(pos);
                for v in &mut nv {
                    if *v > local_idx {
                        *v -= 1;
                    }
                }
                if nv.is_empty() {
                    base_add += c;
                } else {
                    new_terms.push((c, nv));
                }
            } else {
                let mut nv = vars.clone();
                for v in &mut nv {
                    if *v > local_idx {
                        *v -= 1;
                    }
                }
                new_terms.push((*coeff, nv));
            }
        }

        self.base_offset += base_add;
        self.active_terms = new_terms;
        self.solve()
    }

    pub fn solve(&mut self) -> f64 {
        let m = self.free_globals.len();
        if m == 0 {
            return self.base_offset;
        }

        let max_degree = 2 * self.d.max(1);
        let mut residual_terms = Vec::<(f64, Vec<usize>)>::new();
        let mut trivial_extra = 0.0;

        for (coeff, vars) in &self.active_terms {
            if vars.len() <= max_degree {
                residual_terms.push((*coeff, vars.clone()));
            } else {
                trivial_extra += match self.var_type {
                    VarType::Bin => coeff.min(0.0),
                    VarType::Spin => -coeff.abs(),
                };
            }
        }

        if residual_terms.is_empty() {
            log::info!(
                "No active terms of degree ≤ {}, returning trivial bound",
                max_degree
            );
            return self.base_offset + trivial_extra;
        }

        // if m > self.max_vars {
        //     let cheap: f64 = residual_terms
        //         .iter()
        //         .map(|(c, _)| match self.var_type {
        //             VarType::Bin => c.min(0.0),
        //             VarType::Spin => -c.abs(),
        //         })
        //         .sum();
        //     return self.base_offset + trivial_extra + cheap;
        // }

        let apparatus = MomentApparatus::build_for_instance(
            self.d.max(1),
            self.var_type,
            self.max_basis,
            residual_terms.iter().map(|(_, s)| s.as_slice()),
        );

        let mat_dim = apparatus.mat_dim;
        let psd_ncons = mat_dim * (mat_dim + 1) / 2;

        log::info!(
            "Prepared Lasserre relaxation with {} free variables, {} active terms, {} moment vars, {} PSD constraints",
            m,
            residual_terms.len(),
            apparatus.free_moments.len(),
            psd_ncons,
        );

        // Guard: if the basis grew beyond max_basis (shouldn't happen since
        // build_for_instance caps it, but defend anyway) fall back to cheap.
        if mat_dim > self.max_basis {
            let cheap: f64 = residual_terms
                .iter()
                .map(|(c, _)| match self.var_type {
                    VarType::Bin => c.min(0.0),
                    VarType::Spin => -c.abs(),
                })
                .sum();
            return self.base_offset + trivial_extra + cheap;
        }

        let mut obj = vec![0.0; apparatus.free_moments.len()];
        let mut obj_trivial = 0.0;
        for (c, mono) in &residual_terms {
            if let Some(&k) = apparatus.moment_map.get(mono.as_slice()) {
                obj[k] += *c;
            } else {
                obj_trivial += match self.var_type {
                    VarType::Bin => c.min(0.0),
                    VarType::Spin => -c.abs(),
                };
            }
        }

        let base = self.base_offset + trivial_extra + obj_trivial;

        log::info!(
            "Solving Lasserre relaxation with {} free variables, {} active terms, {} moment vars, {} PSD constraints",
            m,
            residual_terms.len(),
            apparatus.free_moments.len(),
            psd_ncons,
        );

        let solve = solve_with_scip(
            base,
            &obj,
            &apparatus,
            self.var_type,
            &self.warm_moments,
            &self.warm_l,
        );

        if let Some(sol) = solve.solution {
            self.warm_moments.clear();
            for (k, mono) in apparatus.free_moments.iter().enumerate() {
                self.warm_moments.insert(mono.clone(), sol.y_vals[k]);
            }
            self.warm_l = sol.l_vals;
        }

        solve.lower_bound
    }
}

#[derive(Debug, Clone)]
struct MomentApparatus {
    #[allow(dead_code)]
    basis: Vec<Vec<usize>>,
    free_moments: Vec<Vec<usize>>,
    moment_map: HashMap<Vec<usize>, usize>,
    m0: Vec<f64>,
    // For each (i,j) entry with i <= j: list of y-indices appearing with +1.
    upper_entry_to_y: Vec<Vec<usize>>,
    mat_dim: usize,
}

/// Maximum basis elements before we stop enriching.  A basis of size B gives
/// a B×B moment matrix with B*(B+1)/2 PSD equality constraints.
pub const MAX_BASIS_ELEMENTS: usize = 36;

/// Don't attempt pairwise cross-term enrichment when the term count exceeds
/// this, to avoid O(m²) overhead on large instances.
const MAX_TERMS_FOR_ENRICHMENT: usize = 400;

/// Sorted set-union of two sorted slices (domain-independent; always union).
fn sorted_union_slices(a: &[usize], b: &[usize]) -> Vec<usize> {
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

/// True iff two sorted variable-index slices share at least one variable.
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

impl MomentApparatus {
    /// Build the moment apparatus for the given hierarchy level `d` and set of
    /// active instance terms (each a sorted slice of local variable indices).
    ///
    /// **Basis construction** at level `d`:
    ///
    /// 1. *Direct*: all sub-monomials of degree ≤ d that are subsets of some
    ///    instance term.  This gives the standard sparse Lasserre basis.
    ///
    /// 2. *Cross-term enrichment* (level ≥ 2): for every pair of terms that
    ///    share at least one variable, add all sub-monomials of degree ≤ d
    ///    from their union.  This captures correlations that live across term
    ///    boundaries — the key improvement over individual-term seeding — and
    ///    is what makes the hierarchy tighten for conflicts between overlapping
    ///    terms.
    ///
    ///    Enrichment stops once the basis reaches [`MAX_BASIS_ELEMENTS`] or
    ///    once the term count exceeds [`MAX_TERMS_FOR_ENRICHMENT`].
    fn build_for_instance<'a, I>(
        d: usize,
        var_type: VarType,
        max_basis: usize,
        instance_terms: I,
    ) -> Self
    where
        I: Iterator<Item = &'a [usize]>,
    {
        let terms: Vec<&'a [usize]> = instance_terms.collect();

        let mut basis_set = BTreeSet::<Vec<usize>>::new();
        basis_set.insert(Vec::new()); // y_∅ = 1 (constant moment)

        // ── Step 1: direct sub-monomials from each individual term ────────────
        // Cap at max_basis so the moment matrix stays tractable for SCIP.
        'direct: for &term in &terms {
            for size in 1..=d.min(term.len()) {
                let mut cur = Vec::new();
                subsets_of_size_from_slice(term, size, 0, &mut cur, &mut |s| {
                    basis_set.insert(s.to_vec());
                });
                if basis_set.len() >= max_basis {
                    break 'direct;
                }
            }
        }

        // ── Step 2: cross-term enrichment for overlapping pairs ───────────────
        // For each pair of terms sharing ≥1 variable, add sub-monomials of
        // their union.  This captures cross-term correlations absent from
        // single-term seeding — e.g. variables that interact only via a shared
        // neighbour term.
        if d >= 2 && basis_set.len() < max_basis && terms.len() <= MAX_TERMS_FOR_ENRICHMENT {
            'pairs: for i in 0..terms.len() {
                for j in (i + 1)..terms.len() {
                    if !terms_share_variable(terms[i], terms[j]) {
                        continue;
                    }
                    let union = sorted_union_slices(terms[i], terms[j]);
                    // Singletons already covered in step 1; start from size 2.
                    for size in 2..=d.min(union.len()) {
                        let mut cur = Vec::new();
                        subsets_of_size_from_slice(&union, size, 0, &mut cur, &mut |s| {
                            basis_set.insert(s.to_vec());
                        });
                        if basis_set.len() >= max_basis {
                            break 'pairs;
                        }
                    }
                }
            }
        }

        let mut basis: Vec<Vec<usize>> = basis_set.into_iter().collect();
        basis.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        let n = basis.len();

        let mut free_set = BTreeSet::<Vec<usize>>::new();
        let mut upper_entry_to_mono = Vec::<Vec<usize>>::with_capacity(n * (n + 1) / 2);

        for i in 0..n {
            for j in i..n {
                let mono = combine_sets(&basis[i], &basis[j], var_type);
                upper_entry_to_mono.push(mono.clone());
                if !mono.is_empty() {
                    free_set.insert(mono);
                }
            }
        }

        let mut free_moments: Vec<Vec<usize>> = free_set.into_iter().collect();
        free_moments.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));

        let mut moment_map = HashMap::new();
        for (k, mono) in free_moments.iter().enumerate() {
            moment_map.insert(mono.clone(), k);
        }

        let mut m0 = vec![0.0; n * n];
        match var_type {
            VarType::Bin => {
                m0[0] = 1.0;
            }
            VarType::Spin => {
                for i in 0..n {
                    m0[i * n + i] = 1.0;
                }
            }
        }

        let mut upper_entry_to_y = Vec::with_capacity(upper_entry_to_mono.len());
        for mono in upper_entry_to_mono {
            if mono.is_empty() {
                upper_entry_to_y.push(Vec::new());
            } else if let Some(&k) = moment_map.get(mono.as_slice()) {
                upper_entry_to_y.push(vec![k]);
            } else {
                upper_entry_to_y.push(Vec::new());
            }
        }

        Self {
            basis,
            free_moments,
            moment_map,
            m0,
            upper_entry_to_y,
            mat_dim: n,
        }
    }

    fn upper_index(&self, i: usize, j: usize) -> usize {
        // Row-major indexing over upper triangle
        i * self.mat_dim - (i * (i - 1)) / 2 + (j - i)
    }

    fn moment_matrix_from_y(&self, y: &[f64]) -> Vec<f64> {
        let n = self.mat_dim;
        let mut mat = self.m0.clone();
        for i in 0..n {
            for j in i..n {
                for &k in &self.upper_entry_to_y[self.upper_index(i, j)] {
                    mat[i * n + j] += y[k];
                    if i != j {
                        mat[j * n + i] += y[k];
                    }
                }
            }
        }
        mat
    }
}

#[derive(Debug, Clone)]
struct ScipSolution {
    y_vals: Vec<f64>,
    l_vals: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ScipSolveOutcome {
    lower_bound: f64,
    solution: Option<ScipSolution>,
}

fn solve_with_scip(
    base: f64,
    obj: &[f64],
    apparatus: &MomentApparatus,
    var_type: VarType,
    warm_moments: &HashMap<Vec<usize>, f64>,
    warm_l: &[f64],
) -> ScipSolveOutcome {
    let trivial: f64 = obj
        .iter()
        .map(|&c| match var_type {
            VarType::Bin => c.min(0.0),
            VarType::Spin => -c.abs(),
        })
        .sum();

    if obj.is_empty() {
        return ScipSolveOutcome {
            lower_bound: base,
            solution: None,
        };
    }

    let mut y_warm = vec![0.0; apparatus.free_moments.len()];
    for (k, mono) in apparatus.free_moments.iter().enumerate() {
        if let Some(v) = warm_moments.get(mono) {
            y_warm[k] = *v;
        }
    }

    let solve_res = solve_with_scip_impl(base, obj, apparatus, var_type, &y_warm, warm_l);
    match solve_res {
        Ok(outcome) => outcome,
        Err(msg) => {
            log::debug!("SCIP SDP solve failed: {msg}");
            ScipSolveOutcome {
                lower_bound: base + trivial,
                solution: None,
            }
        }
    }
}

fn solve_with_scip_impl(
    base: f64,
    obj: &[f64],
    apparatus: &MomentApparatus,
    var_type: VarType,
    y_warm: &[f64],
    warm_l: &[f64],
) -> Result<ScipSolveOutcome, String> {
    unsafe {
        let mut scip: *mut ffi::SCIP = ptr::null_mut();
        scip_call(ffi::SCIPcreate(&mut scip), "SCIPcreate")?;

        let result = (|| {
            scip_call(
                ffi::SCIPincludeDefaultPlugins(scip),
                "SCIPincludeDefaultPlugins",
            )?;

            let prob = CString::new("lasserre_sdp").map_err(|e| e.to_string())?;
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

            let (ylb, yub) = match var_type {
                VarType::Bin => (0.0, 1.0),
                VarType::Spin => (-1.0, 1.0),
            };

            let n = apparatus.mat_dim;
            let d = obj.len();

            let mut y_vars = Vec::<*mut ffi::SCIP_VAR>::with_capacity(d);
            for (k, &coef) in obj.iter().enumerate() {
                let name = CString::new(format!("y_{k}")).map_err(|e| e.to_string())?;
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateVarBasic(
                        scip,
                        &mut v,
                        name.as_ptr(),
                        ylb,
                        yub,
                        coef,
                        ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                    ),
                    "SCIPcreateVarBasic(y)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(y)")?;
                y_vars.push(v);
            }

            let l_bound = (n as f64).sqrt().max(1.0);
            let mut l_vars = Vec::<*mut ffi::SCIP_VAR>::with_capacity(n * (n + 1) / 2);
            for i in 0..n {
                for j in 0..=i {
                    let name = CString::new(format!("l_{i}_{j}")).map_err(|e| e.to_string())?;
                    let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                    scip_call(
                        ffi::SCIPcreateVarBasic(
                            scip,
                            &mut v,
                            name.as_ptr(),
                            -l_bound,
                            l_bound,
                            0.0,
                            ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                        ),
                        "SCIPcreateVarBasic(L)",
                    )?;
                    scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(L)")?;
                    l_vars.push(v);
                }
            }

            // Add quadratic equality constraints: M_ij(y) - sum_r L_ir*L_jr = 0.
            for i in 0..n {
                for j in i..n {
                    let mut lin_vars = Vec::<*mut ffi::SCIP_VAR>::new();
                    let mut lin_coefs = Vec::<f64>::new();
                    for &k in &apparatus.upper_entry_to_y[apparatus.upper_index(i, j)] {
                        lin_vars.push(y_vars[k]);
                        lin_coefs.push(1.0);
                    }

                    let mut qv1 = Vec::<*mut ffi::SCIP_VAR>::new();
                    let mut qv2 = Vec::<*mut ffi::SCIP_VAR>::new();
                    let mut qcoef = Vec::<f64>::new();
                    for r in 0..=i.min(j) {
                        qv1.push(l_vars[tri_index(i, r)]);
                        qv2.push(l_vars[tri_index(j, r)]);
                        qcoef.push(-1.0);
                    }

                    let rhs = -apparatus.m0[i * n + j];
                    let cname = CString::new(format!("psd_{i}_{j}")).map_err(|e| e.to_string())?;
                    let mut cons: *mut ffi::SCIP_CONS = ptr::null_mut();
                    scip_call(
                        ffi::SCIPcreateConsBasicQuadraticNonlinear(
                            scip,
                            &mut cons,
                            cname.as_ptr(),
                            lin_vars.len() as i32,
                            if lin_vars.is_empty() {
                                ptr::null_mut()
                            } else {
                                lin_vars.as_mut_ptr()
                            },
                            if lin_coefs.is_empty() {
                                ptr::null_mut()
                            } else {
                                lin_coefs.as_mut_ptr()
                            },
                            qv1.len() as i32,
                            if qv1.is_empty() {
                                ptr::null_mut()
                            } else {
                                qv1.as_mut_ptr()
                            },
                            if qv2.is_empty() {
                                ptr::null_mut()
                            } else {
                                qv2.as_mut_ptr()
                            },
                            if qcoef.is_empty() {
                                ptr::null_mut()
                            } else {
                                qcoef.as_mut_ptr()
                            },
                            rhs,
                            rhs,
                        ),
                        "SCIPcreateConsBasicQuadraticNonlinear",
                    )?;
                    scip_call(ffi::SCIPaddCons(scip, cons), "SCIPaddCons(psd)")?;
                    let mut crel = cons;
                    scip_call(ffi::SCIPreleaseCons(scip, &mut crel), "SCIPreleaseCons")?;
                }
            }

            // Keep per-node NLP effort bounded to avoid very slow BnB nodes.
            let timelimit_param = CString::new("limits/time").map_err(|e| e.to_string())?;
            // Allow up to 30 s for large matrices (root-node usage);
            // scale proportionally with PSD constraint count.
            let tlim = (0.5 + 0.01 * (n * (n + 1) / 2) as f64).clamp(1.0, 30.0);
            let _ = ffi::SCIPsetRealParam(scip, timelimit_param.as_ptr(), tlim);

            // Warmstart only when there is non-trivial prior information.
            let has_warm_y = y_warm.iter().any(|v| v.abs() > 1e-12);
            let has_warm_l = warm_l.len() == l_vars.len() && warm_l.iter().any(|v| v.abs() > 1e-12);
            if has_warm_y || has_warm_l {
                let mut warm_sol: *mut ffi::SCIP_SOL = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateSol(scip, &mut warm_sol, ptr::null_mut()),
                    "SCIPcreateSol",
                )?;
                for (k, &v) in y_warm.iter().enumerate() {
                    scip_call(
                        ffi::SCIPsetSolVal(scip, warm_sol, y_vars[k], v.clamp(ylb, yub)),
                        "SCIPsetSolVal(y)",
                    )?;
                }

                let l_guess = if warm_l.len() == l_vars.len() {
                    warm_l.to_vec()
                } else {
                    infer_l_from_moments(apparatus, y_warm)
                };
                for (idx, &v) in l_guess.iter().enumerate().take(l_vars.len()) {
                    scip_call(
                        ffi::SCIPsetSolVal(scip, warm_sol, l_vars[idx], v),
                        "SCIPsetSolVal(L)",
                    )?;
                }
                let mut stored: u32 = 0;
                let _ = ffi::SCIPaddSol(scip, warm_sol, &mut stored);
                scip_call(ffi::SCIPfreeSol(scip, &mut warm_sol), "SCIPfreeSol")?;
            }

            scip_call(ffi::SCIPsolve(scip), "SCIPsolve")?;

            let dual = ffi::SCIPgetDualbound(scip);
            let lower_bound = if dual.is_finite() {
                base + dual
            } else {
                base + obj
                    .iter()
                    .map(|&c| match var_type {
                        VarType::Bin => c.min(0.0),
                        VarType::Spin => -c.abs(),
                    })
                    .sum::<f64>()
            };

            let mut solution = None;
            let best = ffi::SCIPgetBestSol(scip);
            if !best.is_null() {
                let mut y_vals = vec![0.0; y_vars.len()];
                for (k, &v) in y_vars.iter().enumerate() {
                    y_vals[k] = ffi::SCIPgetSolVal(scip, best, v);
                }
                let mut l_vals = vec![0.0; l_vars.len()];
                for (k, &v) in l_vars.iter().enumerate() {
                    l_vals[k] = ffi::SCIPgetSolVal(scip, best, v);
                }
                solution = Some(ScipSolution { y_vals, l_vals });
            }

            for v in &mut y_vars {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(y)")?;
            }
            for v in &mut l_vars {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(L)")?;
            }

            Ok(ScipSolveOutcome {
                lower_bound,
                solution,
            })
        })();

        let free_res = scip_call(ffi::SCIPfree(&mut scip), "SCIPfree");
        if let Err(e) = free_res {
            log::debug!("SCIP free failed: {e}");
        }

        result
    }
}

fn infer_l_from_moments(apparatus: &MomentApparatus, y: &[f64]) -> Vec<f64> {
    let n = apparatus.mat_dim;
    let mat = apparatus.moment_matrix_from_y(y);

    // Simple numeric Cholesky-like factorization with jitter.
    let mut l = vec![0.0; n * n];
    let mut a = mat;

    for i in 0..n {
        a[i * n + i] += 1e-9;
    }

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = sum.max(0.0).sqrt();
            } else if l[j * n + j].abs() > 1e-12 {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    let mut tri = vec![0.0; n * (n + 1) / 2];
    for i in 0..n {
        for j in 0..=i {
            tri[tri_index(i, j)] = l[i * n + j];
        }
    }
    tri
}

fn scip_call(code: ffi::SCIP_Retcode, op: &str) -> Result<(), String> {
    if code == ffi::SCIP_Retcode_SCIP_OKAY {
        Ok(())
    } else {
        Err(format!("{op} failed with retcode {code}"))
    }
}

#[inline]
fn tri_index(i: usize, j: usize) -> usize {
    debug_assert!(j <= i);
    i * (i + 1) / 2 + j
}

fn subsets_of_size_from_slice<F: FnMut(&[usize])>(
    data: &[usize],
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    f: &mut F,
) {
    if current.len() == size {
        f(current);
        return;
    }
    let remain = size - current.len();
    if start + remain > data.len() {
        return;
    }
    for i in start..data.len() {
        current.push(data[i]);
        subsets_of_size_from_slice(data, size, i + 1, current, f);
        current.pop();
    }
}

fn combine_sets(a: &[usize], b: &[usize], var_type: VarType) -> Vec<usize> {
    match var_type {
        VarType::Bin => {
            let mut out = Vec::with_capacity(a.len() + b.len());
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Less => {
                        out.push(a[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        out.push(b[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        out.push(a[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            out.extend_from_slice(&a[i..]);
            out.extend_from_slice(&b[j..]);
            out
        }
        VarType::Spin => {
            let mut out = Vec::with_capacity(a.len() + b.len());
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Less => {
                        out.push(a[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        out.push(b[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        i += 1;
                        j += 1;
                    }
                }
            }
            out.extend_from_slice(&a[i..]);
            out.extend_from_slice(&b[j..]);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{domain::Bin, instance::HuboInstance, term::Term};

    fn make_instance(n: usize, terms: Vec<(Vec<usize>, f64)>) -> HuboInstance<f64, Bin> {
        let terms: Vec<Term<f64>> = terms
            .into_iter()
            .map(|(indices, coeff)| Term { indices, coeff })
            .collect();
        HuboInstance::new(n, 0.0, terms)
    }

    #[test]
    fn combine_binary_and_spin() {
        let a = vec![0, 2];
        let b = vec![2, 3];
        assert_eq!(combine_sets(&a, &b, VarType::Bin), vec![0, 2, 3]);
        assert_eq!(combine_sets(&a, &b, VarType::Spin), vec![0, 3]);
    }

    #[test]
    fn basis_is_instance_specific() {
        // Disjoint terms share no variable → no cross-term pairs added.
        let inst = make_instance(5, vec![(vec![0, 1, 2], -1.0), (vec![3, 4], 2.0)]);
        let assignment = vec![None; 5];
        let st = LasserreSdpState::new(&inst, &assignment, 2, 20, 80);

        let app = MomentApparatus::build_for_instance(
            st.d,
            st.var_type,
            st.max_basis,
            st.active_terms.iter().map(|(_, s)| s.as_slice()),
        );

        assert!(app.basis.contains(&vec![]));
        assert!(app.basis.contains(&vec![0]));
        assert!(app.basis.contains(&vec![3, 4]));
        // Variables 0 and 3 never co-appear → cross-term pair must NOT appear.
        assert!(!app.basis.contains(&vec![0, 3]));
    }

    #[test]
    fn level_1_basis_has_only_singletons() {
        let inst = make_instance(4, vec![(vec![0, 1, 2], -1.0), (vec![1, 2, 3], 1.0)]);
        let assignment = vec![None; 4];
        let st = LasserreSdpState::new(&inst, &assignment, 1, 20, 60);

        let app = MomentApparatus::build_for_instance(
            st.d,
            st.var_type,
            st.max_basis,
            st.active_terms.iter().map(|(_, s)| s.as_slice()),
        );

        // At level 1, every basis element has degree ≤ 1.
        assert!(
            app.basis.iter().all(|b| b.len() <= 1),
            "level-1 basis should contain only the constant and singletons, got {:?}",
            app.basis
        );
    }

    #[test]
    fn level_2_basis_larger_than_level_1_for_overlapping_terms() {
        let inst = make_instance(4, vec![(vec![0, 1, 2], -1.0), (vec![1, 2, 3], 1.0)]);
        let assignment = vec![None; 4];

        let app1 = {
            let st = LasserreSdpState::new(&inst, &assignment, 1, 20, 60);
            MomentApparatus::build_for_instance(
                st.d,
                st.var_type,
                st.max_basis,
                st.active_terms.iter().map(|(_, s)| s.as_slice()),
            )
        };
        let app2 = {
            let st = LasserreSdpState::new(&inst, &assignment, 2, 20, 80);
            MomentApparatus::build_for_instance(
                st.d,
                st.var_type,
                st.max_basis,
                st.active_terms.iter().map(|(_, s)| s.as_slice()),
            )
        };

        assert!(
            app2.mat_dim > app1.mat_dim,
            "level-2 basis ({} elements) should be strictly larger than level-1 ({})",
            app2.mat_dim,
            app1.mat_dim
        );
        // Level 2 must include at least one degree-2 monomial.
        assert!(
            app2.basis.iter().any(|b| b.len() == 2),
            "level-2 basis should contain pairwise monomials"
        );
    }

    #[test]
    fn cross_term_enrichment_adds_inter_term_pairs() {
        // Terms {0,1,2} and {2,3,4} overlap at variable 2.
        // At level 2, pairs like {0,3} (crossing the two terms via variable 2)
        // should appear in the basis after enrichment.
        let inst = make_instance(5, vec![(vec![0, 1, 2], -2.0), (vec![2, 3, 4], 1.0)]);
        let assignment = vec![None; 5];
        let st = LasserreSdpState::new(&inst, &assignment, 2, 20, 80);

        let app = MomentApparatus::build_for_instance(
            st.d,
            st.var_type,
            st.max_basis,
            st.active_terms.iter().map(|(_, s)| s.as_slice()),
        );

        // At least one cross-term pair must appear (e.g., {0,3} or {1,4} etc.)
        let has_cross_term_pair = app.basis.iter().any(|b| {
            b.len() == 2 && {
                let (u, v) = (b[0], b[1]);
                // A pair is "cross-term" if one var is in {0,1} and other in {3,4}
                // (they don't share a term directly but are connected via var 2)
                let in_first_only = |x: usize| [0, 1].contains(&x);
                let in_second_only = |x: usize| [3, 4].contains(&x);
                (in_first_only(u) && in_second_only(v)) || (in_second_only(u) && in_first_only(v))
            }
        });
        assert!(
            has_cross_term_pair,
            "cross-term enrichment should add pairs crossing both terms; basis={:?}",
            app.basis
        );
    }

    #[test]
    fn higher_level_basis_contains_higher_degree_monomials() {
        // At level 3, degree-3 sub-monomials of terms should appear.
        let inst = make_instance(4, vec![(vec![0, 1, 2, 3], -1.0)]);
        let assignment = vec![None; 4];

        let app2 = {
            let st = LasserreSdpState::new(&inst, &assignment, 2, 20, 80);
            MomentApparatus::build_for_instance(
                st.d,
                st.var_type,
                st.max_basis,
                st.active_terms.iter().map(|(_, s)| s.as_slice()),
            )
        };
        let app3 = {
            let st = LasserreSdpState::new(&inst, &assignment, 3, 20, 80);
            MomentApparatus::build_for_instance(
                st.d,
                st.var_type,
                st.max_basis,
                st.active_terms.iter().map(|(_, s)| s.as_slice()),
            )
        };

        let max_degree_l2 = app2.basis.iter().map(|b| b.len()).max().unwrap_or(0);
        let max_degree_l3 = app3.basis.iter().map(|b| b.len()).max().unwrap_or(0);
        assert!(
            max_degree_l2 <= 2,
            "level-2 basis degree should be ≤ 2, got {max_degree_l2}"
        );
        assert!(
            max_degree_l3 >= 3,
            "level-3 basis should include degree-3 monomials, max={max_degree_l3}"
        );
    }
}
