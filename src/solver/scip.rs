use std::collections::HashMap;
use std::ffi::CString;
use std::io::{self, Write};
use std::path::Path;

use serde_json;
use std::ptr;

use russcip::Variable;
use russcip::ffi;
use russcip::model::{ProblemCreated, WithSolutions, WithSolvingStats};
use russcip::prelude::*;
use russcip::status::Status;

use crate::Logger;
use crate::coeff::Coeff;
use crate::{domain::{VarDomain, VarType}, instance::HuboInstance};
use crate::solution::BitSolution;

/// Result of solving a HUBO instance.
pub struct SolveResult {
    /// The solver status.
    pub status: Status,
    /// Objective value of the best solution (includes offset), if available.
    pub objective: Option<f64>,
    /// Best bound proven by the solver (includes offset).
    pub best_bound: f64,
    /// Variable assignments in the best solution (indexed by original variable
    /// index), if a feasible solution was found.
    pub solution: Option<BitSolution>,
    /// Solving time in seconds.
    pub solving_time: f64,
    /// Time to solution — wall-clock seconds until the best incumbent was found.
    pub tts: Option<f64>,
    /// Number of branch-and-bound nodes explored.
    pub n_nodes: usize,
}

impl SolveResult {
    /// Write the solution to a file.
    ///
    /// Format:
    /// ```text
    /// # HUES solution file
    /// STATUS <status>
    /// OBJECTIVE <value>
    /// BEST_BOUND <value>
    /// TIME <seconds>
    /// NODES <count>
    /// SOLUTION
    /// x0 = 1
    /// x1 = 0
    /// ...
    /// ```
    pub fn write_solution_file(&self, path: impl AsRef<Path>, var_type: VarType) -> io::Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let sol_arr = self.solution.as_ref().map(|s| s.to_json_array(var_type));
            let obj = serde_json::json!({
                "status": format!("{:?}", self.status),
                "objective": self.objective,
                "best_bound": self.best_bound,
                "time_s": self.solving_time,
                "tts_s": self.tts,
                "nodes": self.n_nodes,
                "solution": sol_arr,
            });
            return std::fs::write(path, serde_json::to_string_pretty(&obj).unwrap());
        }

        let mut f = std::fs::File::create(path)?;

        writeln!(f, "# HUES solution file")?;
        writeln!(f, "STATUS {:?}", self.status)?;

        if let Some(obj) = self.objective {
            writeln!(f, "OBJECTIVE {obj}")?;
        } else {
            writeln!(f, "OBJECTIVE n/a")?;
        }

        writeln!(f, "BEST_BOUND {}", self.best_bound)?;
        writeln!(f, "TIME {:.6}", self.solving_time)?;
        if let Some(tts) = self.tts {
            writeln!(f, "TTS {tts:.6}")?;
        }
        writeln!(f, "NODES {}", self.n_nodes)?;

        if let Some(ref sol) = self.solution {
            writeln!(f, "SOLUTION")?;
            sol.write_to(&mut f, var_type)?;
        }

        Ok(())
    }
}

/// Solver configuration.
pub struct SolverConfig {
    pub time_limit: Option<usize>,
    pub node_limit: Option<i64>,
    pub gap_limit: Option<f64>,
    pub verbosity: i32,
    pub threads: Option<i32>,
    /// If set, the solution will be written to this file after solving.
    pub solution_file: Option<String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            gap_limit: None,
            verbosity: 4,
            threads: None,
            solution_file: None,
        }
    }
}

/// Get (or create) a linearised product variable for all_vars[a] * all_vars[b].
///
/// Introduces auxiliary binary variable w with McCormick constraints:
///   w <= all_vars[a],  w <= all_vars[b],  w >= all_vars[a] + all_vars[b] - 1
///
/// Returns the index of the product variable in `all_vars`.
fn get_or_create_product(
    model: &mut Model<ProblemCreated>,
    a: usize,
    b: usize,
    all_vars: &mut Vec<Variable>,
    product_cache: &mut HashMap<(usize, usize), usize>,
    aux_count: &mut usize,
) -> usize {
    let key = if a <= b { (a, b) } else { (b, a) };
    if let Some(&idx) = product_cache.get(&key) {
        return idx;
    }

    let name = format!("w{}", *aux_count);
    *aux_count += 1;
    let w = model.add_var(0.0, 1.0, 0.0, &name, russcip::VarType::Binary);

    // w <= all_vars[a]  =>  w - all_vars[a] <= 0
    model.add_cons(
        vec![&w, &all_vars[key.0]],
        &[1.0, -1.0],
        f64::NEG_INFINITY,
        0.0,
        &format!("{name}_le_a"),
    );
    // w <= all_vars[b]  =>  w - all_vars[b] <= 0
    model.add_cons(
        vec![&w, &all_vars[key.1]],
        &[1.0, -1.0],
        f64::NEG_INFINITY,
        0.0,
        &format!("{name}_le_b"),
    );
    // w >= all_vars[a] + all_vars[b] - 1  =>  w - all_vars[a] - all_vars[b] >= -1
    model.add_cons(
        vec![&w, &all_vars[key.0], &all_vars[key.1]],
        &[1.0, -1.0, -1.0],
        -1.0,
        f64::INFINITY,
        &format!("{name}_ge"),
    );

    let idx = all_vars.len();
    all_vars.push(w);
    product_cache.insert(key, idx);
    idx
}

/// Linearise a product of binary variables and accumulate into objective
/// coefficients.
///
/// Higher-degree products are reduced iteratively:
///   x_a * x_b * x_c  ->  w_{a,b} * x_c  ->  w_{(a,b),c}
#[allow(clippy::too_many_arguments)]
fn linearise_product(
    indices: &[usize],
    coeff: f64,
    model: &mut Model<ProblemCreated>,
    product_cache: &mut HashMap<(usize, usize), usize>,
    aux_count: &mut usize,
    obj_coefs: &mut HashMap<usize, f64>,
    all_vars: &mut Vec<Variable>,
    constant: &mut f64,
) {
    match indices.len() {
        0 => {
            *constant += coeff;
        }
        1 => {
            *obj_coefs.entry(indices[0]).or_insert(0.0) += coeff;
        }
        _ => {
            // Iteratively reduce the product.
            let mut current_idx = indices[0];
            for &next_idx in &indices[1..] {
                current_idx = get_or_create_product(
                    model,
                    current_idx,
                    next_idx,
                    all_vars,
                    product_cache,
                    aux_count,
                );
            }
            *obj_coefs.entry(current_idx).or_insert(0.0) += coeff;
        }
    }
}

/// Solve a HUBO instance using SCIP.
///
/// The HUBO polynomial objective is linearised into a mixed-integer program:
///
/// - For `BIN` variables (x_i in {0,1}), products of binary variables are
///   reduced to linear terms by introducing auxiliary variables.
/// - For `SPIN` variables (s_i in {-1,+1}), we substitute s_i = 2x_i - 1
///   and expand, then linearise the resulting binary polynomial.
pub fn solve<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &SolverConfig,
    logger: &Logger,
    init_solution: Option<Vec<C>>,
) -> SolveResult {
    // ---- Build SCIP model --------------------------------------------------
    let _ = logger;
    log::info!(
        "building SCIP model for {} variables, {} terms",
        instance.n_vars(),
        instance.n_terms()
    );

    let mut model = Model::new()
        .set_display_verbosity(config.verbosity)
        .include_default_plugins()
        .create_prob("hubo")
        .minimize();

    // Apply solver parameters (each consumes and returns model).
    if let Some(tl) = config.time_limit {
        model = model.set_time_limit(tl);
    }
    if let Some(nl) = config.node_limit {
        model = model.set_longint_param("limits/nodes", nl).unwrap();
    }
    if let Some(gap) = config.gap_limit {
        model = model.set_real_param("limits/gap", gap).unwrap();
    }
    if let Some(threads) = config.threads {
        model = model
            .set_int_param("parallel/maxnthreads", threads)
            .unwrap();
    }

    // ---- Create binary decision variables ----------------------------------
    let n = instance.n_vars();
    let mut all_vars: Vec<Variable> = Vec::with_capacity(n);
    for i in 0..n {
        let v = model.add_var(0.0, 1.0, 0.0, &format!("x{i}"), russcip::VarType::Binary);
        all_vars.push(v);
    }
    log::debug!("created {} binary decision variables", n);

    // ---- Linearise the polynomial objective --------------------------------
    let mut product_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut aux_count: usize = 0;
    let mut obj_coefs: HashMap<usize, f64> = HashMap::new();
    let mut constant = instance.offset.to_f64();

    for term in &instance.terms {
        let coeff = term.coeff.to_f64();

        match V::VAR_TYPE {
            VarType::Bin => {
                linearise_product(
                    &term.indices,
                    coeff,
                    &mut model,
                    &mut product_cache,
                    &mut aux_count,
                    &mut obj_coefs,
                    &mut all_vars,
                    &mut constant,
                );
            }
            VarType::Spin => {
                // Expand s_i = 2x_i - 1 for each variable in the monomial.
                //   prod(2x_i - 1) = sum over subsets S:
                //       (-1)^(k-|S|) * 2^|S| * prod(x_i for i in S)
                let k = term.indices.len();
                for mask in 0..(1u64 << k) {
                    let subset_size = mask.count_ones() as usize;
                    let sign = if (k - subset_size).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    };
                    let factor = sign * (1u64 << subset_size) as f64;
                    let sub_coeff = coeff * factor;

                    let mut subset_indices: Vec<usize> = Vec::new();
                    for bit in 0..k {
                        if mask & (1u64 << bit) != 0 {
                            subset_indices.push(term.indices[bit]);
                        }
                    }

                    linearise_product(
                        &subset_indices,
                        sub_coeff,
                        &mut model,
                        &mut product_cache,
                        &mut aux_count,
                        &mut obj_coefs,
                        &mut all_vars,
                        &mut constant,
                    );
                }
            }
        }
    }

    // ---- Encode objective via a free continuous variable Z ------------------
    // We minimise Z subject to Z = sum(c_i * v_i) + constant.
    let z = model.add_var(
        f64::NEG_INFINITY,
        f64::INFINITY,
        1.0,
        "Z",
        russcip::VarType::Continuous,
    );

    let mut cons_vars: Vec<&Variable> = Vec::with_capacity(obj_coefs.len() + 1);
    let mut cons_coefs: Vec<f64> = Vec::with_capacity(obj_coefs.len() + 1);

    cons_vars.push(&z);
    cons_coefs.push(1.0);

    for (&var_idx, &c) in &obj_coefs {
        cons_vars.push(&all_vars[var_idx]);
        cons_coefs.push(-c);
    }

    model.add_cons(cons_vars, &cons_coefs, constant, constant, "objective_def");

    log::info!(
        "linearisation complete: {} total variables ({} auxiliary), constant={}",
        all_vars.len(),
        aux_count,
        constant
    );

    // ---- Inject initial solution if provided --------------------------------
    if let Some(ref init_vals) = init_solution {
        log::info!("injecting initial solution as warm-start hint");
        let sol = model.create_orig_sol();

        // 1) Set the original decision variables.
        let mut bin_vals = vec![0.0_f64; all_vars.len()];
        for i in 0..n {
            let bin_val = match V::VAR_TYPE {
                VarType::Bin => init_vals[i].to_f64(),
                // s_i = 2x_i - 1  =>  x_i = (s_i + 1) / 2
                VarType::Spin => (init_vals[i].to_f64() + 1.0) / 2.0,
            };
            bin_vals[i] = bin_val;
            sol.set_val(&all_vars[i], bin_val);
        }

        // 2) Set the auxiliary product variables: w_{a,b} = x_a * x_b.
        //    Sort by output index so chained products (w depending on earlier w)
        //    are evaluated in the correct order.
        let mut products: Vec<_> = product_cache.iter().collect();
        products.sort_by_key(|&(_, &idx)| idx);
        for &(&(a, b), &idx) in &products {
            let prod = (bin_vals[a] * bin_vals[b]).round();
            bin_vals[idx] = prod;
            sol.set_val(&all_vars[idx], prod);
        }

        // 3) Set Z = sum(c_i * v_i) + constant.
        let z_val: f64 = obj_coefs
            .iter()
            .map(|(&i, &c)| c * bin_vals[i])
            .sum::<f64>()
            + constant;
        sol.set_val(&z, z_val);

        match model.add_sol(sol) {
            Ok(()) => {
                log::info!("initial solution accepted by SCIP");
            }
            Err(e) => {
                log::warn!("initial solution rejected by SCIP: {:?}", e);
            }
        }
    }

    // ---- Solve -------------------------------------------------------------
    log::info!("starting SCIP solver");
    let solved = model.solve();

    let status = solved.status();
    let solving_time = solved.solving_time();
    let n_nodes = solved.n_nodes();
    let best_bound = solved.best_bound();

    log::info!(
        "solver finished: status={:?}, time={:.3}s, nodes={}",
        status,
        solving_time,
        n_nodes
    );
    log::info!("best bound = {}", best_bound);

    // Try to extract the best feasible solution regardless of status.
    // SCIP's best_sol() returns None when no feasible solution was found.
    let (objective, solution, tts) = if let Some(sol) = solved.best_sol() {
        let tts = unsafe { ffi::SCIPsolGetTime(sol.inner()) };
        log::info!("tts = {:.3}s", tts);
        // SCIP uses binary x_i ∈ {0,1} for all variable types.
        // For SPIN, the encoding is s_i = 2x_i - 1, so x_i = 1 ↔ s_i = +1 (high).
        let mut bitsol = BitSolution::new(n);
        for (i, var) in all_vars.iter().take(n).enumerate() {
            bitsol.values.set(i, sol.val(var) >= 0.5);
        }
        let obj = bitsol.evaluate(instance).to_f64();
        log::info!("objective = {}", obj);
        log::debug!("solution extracted for {} variables", n);
        (Some(obj), Some(bitsol), Some(tts))
    } else {
        log::warn!("no feasible solution found");
        (None, None, None)
    };

    let result = SolveResult {
        status,
        objective,
        best_bound,
        solution,
        solving_time,
        tts,
        n_nodes,
    };

    // Auto-write solution file if configured.
    if let Some(ref path) = config.solution_file {
        match result.write_solution_file(path, V::VAR_TYPE) {
            Ok(()) => log::info!("solution written to {}", path),
            Err(e) => log::error!("failed to write solution file {}: {}", path, e),
        }
    }

    result
}

/// Explicit name for the McCormick-based linearization path.
pub fn solve_mccormick<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &SolverConfig,
    logger: &Logger,
    init_solution: Option<Vec<C>>,
) -> SolveResult {
    solve(instance, config, logger, init_solution)
}

unsafe fn create_var_expr(
    scip: *mut ffi::SCIP,
    var: *mut ffi::SCIP_VAR,
) -> Result<*mut ffi::SCIP_EXPR, String> {
    let mut expr: *mut ffi::SCIP_EXPR = ptr::null_mut();
    scip_call(
        unsafe { ffi::SCIPcreateExprVar(scip, &mut expr, var, None, ptr::null_mut()) },
        "SCIPcreateExprVar",
    )?;
    Ok(expr)
}

unsafe fn create_spin_factor_expr(
    scip: *mut ffi::SCIP,
    var: *mut ffi::SCIP_VAR,
) -> Result<*mut ffi::SCIP_EXPR, String> {
    let mut child = unsafe { create_var_expr(scip, var)? };

    let mut children = [child];
    let mut coefs = [2.0_f64];
    let mut expr: *mut ffi::SCIP_EXPR = ptr::null_mut();
    unsafe {
        scip_call(
            ffi::SCIPcreateExprSum(
                scip,
                &mut expr,
                1,
                children.as_mut_ptr(),
                coefs.as_mut_ptr(),
                -1.0,
                None,
                ptr::null_mut(),
            ),
            "SCIPcreateExprSum(2x-1)",
        )?;
        scip_call(
            ffi::SCIPreleaseExpr(scip, &mut child),
            "SCIPreleaseExpr(var)",
        )
    }?;
    Ok(expr)
}

/// Solve a HUBO instance with SCIP by representing each multilinear term via
/// one full nonlinear expression constraint added through SCIP FFI and an
/// explicit epigraph variable `Z`.
pub fn solve_nonlinear_constraint<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    config: &SolverConfig,
    logger: &Logger,
    init_solution: Option<Vec<C>>,
) -> SolveResult {
    let _ = logger;
    log::info!(
        "building SCIP FFI nonlinear model for {} variables, {} terms",
        instance.n_vars(),
        instance.n_terms()
    );

    let result = unsafe {
        let mut scip: *mut ffi::SCIP = ptr::null_mut();
        let run = (|| -> Result<SolveResult, String> {
            scip_call(ffi::SCIPcreate(&mut scip), "SCIPcreate")?;
            scip_call(
                ffi::SCIPincludeDefaultPlugins(scip),
                "SCIPincludeDefaultPlugins",
            )?;

            let prob = CString::new("hubo_nonlinear_ffi").map_err(|e| e.to_string())?;
            scip_call(
                ffi::SCIPcreateProbBasic(scip, prob.as_ptr()),
                "SCIPcreateProbBasic",
            )?;
            scip_call(
                ffi::SCIPsetObjsense(scip, ffi::SCIP_Objsense_SCIP_OBJSENSE_MINIMIZE),
                "SCIPsetObjsense",
            )?;

            let vparam = CString::new("display/verblevel").map_err(|e| e.to_string())?;
            let _ = ffi::SCIPsetIntParam(scip, vparam.as_ptr(), config.verbosity);

            if let Some(tl) = config.time_limit {
                let pname = CString::new("limits/time").map_err(|e| e.to_string())?;
                let _ = ffi::SCIPsetRealParam(scip, pname.as_ptr(), tl as f64);
            }
            if let Some(nl) = config.node_limit {
                let pname = CString::new("limits/nodes").map_err(|e| e.to_string())?;
                let _ = ffi::SCIPsetLongintParam(scip, pname.as_ptr(), nl);
            }
            if let Some(gap) = config.gap_limit {
                let pname = CString::new("limits/gap").map_err(|e| e.to_string())?;
                let _ = ffi::SCIPsetRealParam(scip, pname.as_ptr(), gap);
            }
            if let Some(threads) = config.threads {
                let pname = CString::new("parallel/maxnthreads").map_err(|e| e.to_string())?;
                let _ = ffi::SCIPsetIntParam(scip, pname.as_ptr(), threads);
            }

            let n = instance.n_vars();
            let mut x_vars = Vec::<*mut ffi::SCIP_VAR>::with_capacity(n);
            for i in 0..n {
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                let name = CString::new(format!("x{i}")).map_err(|e| e.to_string())?;
                scip_call(
                    ffi::SCIPcreateVarBasic(
                        scip,
                        &mut v,
                        name.as_ptr(),
                        0.0,
                        1.0,
                        0.0,
                        ffi::SCIP_Vartype_SCIP_VARTYPE_BINARY,
                    ),
                    "SCIPcreateVarBasic(x)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(x)")?;
                x_vars.push(v);
            }

            let mut z_var: *mut ffi::SCIP_VAR = ptr::null_mut();
            let z_name = CString::new("Z").map_err(|e| e.to_string())?;
            scip_call(
                ffi::SCIPcreateVarBasic(
                    scip,
                    &mut z_var,
                    z_name.as_ptr(),
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    1.0,
                    ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                ),
                "SCIPcreateVarBasic(Z)",
            )?;
            scip_call(ffi::SCIPaddVar(scip, z_var), "SCIPaddVar(Z)")?;

            let mut constant = instance.offset.to_f64();
            let mut term_exprs = Vec::<*mut ffi::SCIP_EXPR>::new();
            for term in &instance.terms {
                if term.indices.is_empty() {
                    constant += term.coeff.to_f64();
                    continue;
                }

                let mut factors = Vec::<*mut ffi::SCIP_EXPR>::with_capacity(term.indices.len());
                for &idx in &term.indices {
                    let factor = match V::VAR_TYPE {
                        VarType::Bin => create_var_expr(scip, x_vars[idx])?,
                        VarType::Spin => create_spin_factor_expr(scip, x_vars[idx])?,
                    };
                    factors.push(factor);
                }

                let mut term_expr: *mut ffi::SCIP_EXPR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateExprProduct(
                        scip,
                        &mut term_expr,
                        factors.len() as i32,
                        factors.as_mut_ptr(),
                        term.coeff.to_f64(),
                        None,
                        ptr::null_mut(),
                    ),
                    "SCIPcreateExprProduct(term)",
                )?;
                for expr in &mut factors {
                    scip_call(ffi::SCIPreleaseExpr(scip, expr), "SCIPreleaseExpr(factor)")?;
                }
                term_exprs.push(term_expr);
            }

            let z_expr = create_var_expr(scip, z_var)?;
            term_exprs.push(z_expr);

            let mut sum_coefs = vec![1.0_f64; term_exprs.len()];
            if let Some(last) = sum_coefs.last_mut() {
                *last = -1.0;
            }

            let mut root_expr: *mut ffi::SCIP_EXPR = ptr::null_mut();
            scip_call(
                ffi::SCIPcreateExprSum(
                    scip,
                    &mut root_expr,
                    term_exprs.len() as i32,
                    term_exprs.as_mut_ptr(),
                    sum_coefs.as_mut_ptr(),
                    constant,
                    None,
                    ptr::null_mut(),
                ),
                "SCIPcreateExprSum(objective)",
            )?;
            for expr in &mut term_exprs {
                scip_call(ffi::SCIPreleaseExpr(scip, expr), "SCIPreleaseExpr(term)")?;
            }

            let cname = CString::new("objective_def_nl").map_err(|e| e.to_string())?;
            let mut cons: *mut ffi::SCIP_CONS = ptr::null_mut();
            scip_call(
                ffi::SCIPcreateConsBasicNonlinear(
                    scip,
                    &mut cons,
                    cname.as_ptr(),
                    root_expr,
                    0.0,
                    0.0,
                ),
                "SCIPcreateConsBasicNonlinear",
            )?;
            scip_call(ffi::SCIPaddCons(scip, cons), "SCIPaddCons(nonlinear)")?;
            scip_call(ffi::SCIPreleaseCons(scip, &mut cons), "SCIPreleaseCons")?;
            scip_call(
                ffi::SCIPreleaseExpr(scip, &mut root_expr),
                "SCIPreleaseExpr(root)",
            )?;

            if let Some(ref init_vals) = init_solution {
                let mut warm_sol: *mut ffi::SCIP_SOL = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateSol(scip, &mut warm_sol, ptr::null_mut()),
                    "SCIPcreateSol",
                )?;

                for i in 0..n {
                    let x = match V::VAR_TYPE {
                        VarType::Bin => init_vals[i].to_f64(),
                        VarType::Spin => (init_vals[i].to_f64() + 1.0) / 2.0,
                    };
                    scip_call(
                        ffi::SCIPsetSolVal(scip, warm_sol, x_vars[i], x),
                        "SCIPsetSolVal(x)",
                    )?;
                }
                let z_guess = instance.evaluate(init_vals).to_f64();
                scip_call(
                    ffi::SCIPsetSolVal(scip, warm_sol, z_var, z_guess),
                    "SCIPsetSolVal(Z)",
                )?;

                let mut stored: u32 = 0;
                let _ = ffi::SCIPaddSol(scip, warm_sol, &mut stored);
                scip_call(ffi::SCIPfreeSol(scip, &mut warm_sol), "SCIPfreeSol")?;
            }

            scip_call(ffi::SCIPsolve(scip), "SCIPsolve")?;

            let status = Status::from(ffi::SCIPgetStatus(scip));
            let solving_time = ffi::SCIPgetSolvingTime(scip);
            let n_nodes = ffi::SCIPgetNNodes(scip) as usize;
            let best_bound = ffi::SCIPgetDualbound(scip);

            let best = ffi::SCIPgetBestSol(scip);
            let (objective, solution, tts) = if !best.is_null() {
                let tts = ffi::SCIPsolGetTime(best);
                let mut bitsol = BitSolution::new(n);
                for (i, &v) in x_vars.iter().enumerate() {
                    bitsol
                        .values
                        .set(i, ffi::SCIPgetSolVal(scip, best, v) >= 0.5);
                }
                let obj = bitsol.evaluate(instance).to_f64();
                (Some(obj), Some(bitsol), Some(tts))
            } else {
                (None, None, None)
            };

            for v in &mut x_vars {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(x)")?;
            }
            scip_call(ffi::SCIPreleaseVar(scip, &mut z_var), "SCIPreleaseVar(Z)")?;

            Ok(SolveResult {
                status,
                objective,
                best_bound,
                solution,
                solving_time,
                tts,
                n_nodes,
            })
        })();

        let free_res = if scip.is_null() {
            Ok(())
        } else {
            scip_call(ffi::SCIPfree(&mut scip), "SCIPfree")
        };
        if let Err(e) = free_res {
            log::debug!("SCIP free failed: {e}");
        }

        match run {
            Ok(res) => res,
            Err(msg) => {
                log::error!("FFI nonlinear SCIP solve failed: {msg}");
                SolveResult {
                    status: Status::Unknown,
                    objective: None,
                    best_bound: f64::NAN,
                    solution: None,
                    solving_time: 0.0,
                    tts: None,
                    n_nodes: 0,
                }
            }
        }
    };

    if let Some(ref path) = config.solution_file {
        match result.write_solution_file(path, V::VAR_TYPE) {
            Ok(()) => log::info!("solution written to {}", path),
            Err(e) => log::error!("failed to write solution file {}: {}", path, e),
        }
    }

    result
}

fn scip_call(code: ffi::SCIP_Retcode, op: &str) -> Result<(), String> {
    if code == ffi::SCIP_Retcode_SCIP_OKAY {
        Ok(())
    } else {
        Err(format!("{op} failed with retcode {code}"))
    }
}
