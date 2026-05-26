use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::ptr;

use russcip::ffi;

use crate::coeff::Coeff;
use crate::lasserre::LasserreConfig;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
};

#[derive(Debug, Clone)]
pub struct ChordalSdpDecomposition {
    pub n_vars: usize,
    pub peo: Vec<usize>,
    pub fill_edges: Vec<(usize, usize)>,
    pub maximal_cliques: Vec<Vec<usize>>,
    pub rip_holds: bool,
}

#[derive(Debug, Clone)]
struct ReducedQuadratic {
    var_type: VarType,
    n_free: usize,
    base_offset: f64,
    linear: Vec<f64>,
    quad: HashMap<(usize, usize), f64>,
    higher_order_trivial: f64,
}

/// Build interaction graph -> chordal completion (min-fill) -> PEO -> maximal
/// cliques -> RIP check, then solve a block-diagonal first-order SDP with SCIP.
///
/// Notes:
/// - This is currently a first-order sparse SDP path (`order == 1`).
/// - For `order > 1`, this function falls back to the dense Lasserre routine.
pub fn chordal_lasserre_lower_bound<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
    cfg: &LasserreConfig,
) -> f64 {
    if cfg.level.max(1) != 1 {
        return crate::lasserre::lasserre_lower_bound(instance, assignment, cfg);
    }

    let reduced = reduce_to_quadratic(instance, assignment);
    if reduced.n_free == 0 {
        return reduced.base_offset;
    }

    let cheap_quad = reduced
        .linear
        .iter()
        .map(|&c| termwise_min(c, reduced.var_type))
        .sum::<f64>()
        + reduced
            .quad
            .values()
            .map(|&c| termwise_min(c, reduced.var_type))
            .sum::<f64>();

    if reduced.n_free > cfg.max_vars {
        return reduced.base_offset + reduced.higher_order_trivial + cheap_quad;
    }

    let (completed_adj, peo, fill_edges) = min_fill_completion(reduced.n_free, &reduced.quad);
    let maximal_cliques = maximal_cliques_from_peo(&completed_adj, &peo);
    let rip_holds = running_intersection_holds(&maximal_cliques);

    let decomp = ChordalSdpDecomposition {
        n_vars: reduced.n_free,
        peo,
        fill_edges,
        maximal_cliques,
        rip_holds,
    };

    let base = reduced.base_offset + reduced.higher_order_trivial;
    match solve_block_sdp_with_scip(base, &reduced, &decomp) {
        Ok(v) => v,
        Err(msg) => {
            log::debug!("Chordal SDP solve failed: {msg}");
            base + cheap_quad
        }
    }
}

fn reduce_to_quadratic<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    assignment: &[Option<C>],
) -> ReducedQuadratic {
    let var_type = V::VAR_TYPE;

    let mut g2l = vec![usize::MAX; instance.n_vars()];
    let mut n_free = 0usize;
    for (g, slot) in assignment.iter().enumerate() {
        if slot.is_none() {
            g2l[g] = n_free;
            n_free += 1;
        }
    }

    let mut base_offset = instance.offset.to_f64();
    let mut linear = vec![0.0; n_free];
    let mut quad = HashMap::<(usize, usize), f64>::new();
    let mut higher_order_trivial = 0.0;

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

        match locals.len() {
            0 => {
                base_offset += coeff;
            }
            1 => {
                linear[locals[0]] += coeff;
            }
            2 => {
                let key = if locals[0] < locals[1] {
                    (locals[0], locals[1])
                } else {
                    (locals[1], locals[0])
                };
                *quad.entry(key).or_insert(0.0) += coeff;
            }
            _ => {
                higher_order_trivial += termwise_min(coeff, var_type);
            }
        }
    }

    quad.retain(|_, c| c.abs() > 1e-12);

    ReducedQuadratic {
        var_type,
        n_free,
        base_offset,
        linear,
        quad,
        higher_order_trivial,
    }
}

#[inline]
fn termwise_min(coeff: f64, var_type: VarType) -> f64 {
    match var_type {
        VarType::Bin => coeff.min(0.0),
        VarType::Spin => -coeff.abs(),
    }
}

fn min_fill_completion(
    n: usize,
    quad: &HashMap<(usize, usize), f64>,
) -> (Vec<Vec<bool>>, Vec<usize>, Vec<(usize, usize)>) {
    let mut adj = vec![vec![false; n]; n];
    for &(u, v) in quad.keys() {
        adj[u][v] = true;
        adj[v][u] = true;
    }

    let mut alive = vec![true; n];
    let mut left = n;
    let mut order = Vec::with_capacity(n);
    let mut fill_edges = Vec::<(usize, usize)>::new();

    while left > 0 {
        let mut best_v = None;
        let mut best_fill = usize::MAX;
        let mut best_deg = usize::MAX;

        for v in 0..n {
            if !alive[v] {
                continue;
            }
            let neigh: Vec<usize> = (0..n).filter(|&u| alive[u] && adj[v][u]).collect();
            let deg = neigh.len();
            let mut fill = 0usize;
            for i in 0..neigh.len() {
                for j in (i + 1)..neigh.len() {
                    if !adj[neigh[i]][neigh[j]] {
                        fill += 1;
                    }
                }
            }
            if fill < best_fill || (fill == best_fill && deg < best_deg) {
                best_fill = fill;
                best_deg = deg;
                best_v = Some(v);
            }
        }

        let v = best_v.expect("alive vertex must exist");
        let neigh: Vec<usize> = (0..n).filter(|&u| alive[u] && adj[v][u]).collect();

        for i in 0..neigh.len() {
            for j in (i + 1)..neigh.len() {
                let a = neigh[i];
                let b = neigh[j];
                if !adj[a][b] {
                    adj[a][b] = true;
                    adj[b][a] = true;
                    let e = if a < b { (a, b) } else { (b, a) };
                    fill_edges.push(e);
                }
            }
        }

        alive[v] = false;
        left -= 1;
        order.push(v);
    }

    fill_edges.sort_unstable();
    fill_edges.dedup();

    (adj, order, fill_edges)
}

fn maximal_cliques_from_peo(adj: &[Vec<bool>], peo: &[usize]) -> Vec<Vec<usize>> {
    let n = peo.len();
    let mut pos = vec![0usize; n];
    for (i, &v) in peo.iter().enumerate() {
        pos[v] = i;
    }

    let mut candidates: Vec<Vec<usize>> = Vec::new();
    for (i, &v) in peo.iter().enumerate() {
        let mut clique = vec![v];
        for u in 0..n {
            if u != v && adj[v][u] && pos[u] > i {
                clique.push(u);
            }
        }
        clique.sort_unstable();
        candidates.push(clique);
    }

    candidates.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

    let mut maximal = Vec::<Vec<usize>>::new();
    'outer: for c in candidates {
        for m in &maximal {
            if is_subset(&c, m) {
                continue 'outer;
            }
        }
        maximal.push(c);
    }

    maximal.sort();
    maximal
}

fn is_subset(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            return false;
        }
    }
    i == a.len()
}

fn running_intersection_holds(cliques: &[Vec<usize>]) -> bool {
    if cliques.len() <= 1 {
        return true;
    }

    // Build a maximum-weight clique tree on intersections.
    let mut edges = Vec::<(usize, usize, usize)>::new();
    for i in 0..cliques.len() {
        for j in (i + 1)..cliques.len() {
            let w = intersection_size(&cliques[i], &cliques[j]);
            if w > 0 {
                edges.push((w, i, j));
            }
        }
    }
    edges.sort_by_key(|b| std::cmp::Reverse(b.0));

    let mut dsu = Dsu::new(cliques.len());
    let mut tree = vec![Vec::<usize>::new(); cliques.len()];
    for (_, i, j) in edges {
        if dsu.union(i, j) {
            tree[i].push(j);
            tree[j].push(i);
        }
    }

    // Build variable -> cliques map.
    let mut var_to_cliques = HashMap::<usize, Vec<usize>>::new();
    for (ci, clique) in cliques.iter().enumerate() {
        for &v in clique {
            var_to_cliques.entry(v).or_default().push(ci);
        }
    }

    // RIP: for each variable, cliques containing it must be connected in clique tree.
    for owners in var_to_cliques.values() {
        if owners.len() <= 1 {
            continue;
        }
        let owner_set: BTreeSet<usize> = owners.iter().copied().collect();
        let start = owners[0];
        let mut stack = vec![start];
        let mut seen = vec![false; cliques.len()];
        seen[start] = true;
        let mut reached = BTreeSet::<usize>::new();

        while let Some(c) = stack.pop() {
            if owner_set.contains(&c) {
                reached.insert(c);
            }
            for &nxt in &tree[c] {
                if !seen[nxt] {
                    seen[nxt] = true;
                    stack.push(nxt);
                }
            }
        }

        if reached.len() != owner_set.len() {
            return false;
        }
    }

    true
}

fn intersection_size(a: &[usize], b: &[usize]) -> usize {
    let (mut i, mut j, mut s) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            s += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    s
}

#[derive(Debug, Clone)]
struct Dsu {
    p: Vec<usize>,
    r: Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            p: (0..n).collect(),
            r: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.p[x] != x {
            let root = self.find(self.p[x]);
            self.p[x] = root;
        }
        self.p[x]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.r[ra] < self.r[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.p[rb] = ra;
        if self.r[ra] == self.r[rb] {
            self.r[ra] += 1;
        }
        true
    }
}

fn solve_block_sdp_with_scip(
    base: f64,
    reduced: &ReducedQuadratic,
    decomp: &ChordalSdpDecomposition,
) -> Result<f64, String> {
    unsafe {
        let mut scip: *mut ffi::SCIP = ptr::null_mut();
        scip_call(ffi::SCIPcreate(&mut scip), "SCIPcreate")?;

        let result = (|| {
            scip_call(
                ffi::SCIPincludeDefaultPlugins(scip),
                "SCIPincludeDefaultPlugins",
            )?;

            let prob = CString::new("chordal_sdp").map_err(|e| e.to_string())?;
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

            let (ylb, yub) = match reduced.var_type {
                VarType::Bin => (0.0, 1.0),
                VarType::Spin => (-1.0, 1.0),
            };

            let mut y1_vars = Vec::<*mut ffi::SCIP_VAR>::with_capacity(reduced.n_free);
            for i in 0..reduced.n_free {
                let name = CString::new(format!("y_{i}")).map_err(|e| e.to_string())?;
                let mut v: *mut ffi::SCIP_VAR = ptr::null_mut();
                scip_call(
                    ffi::SCIPcreateVarBasic(
                        scip,
                        &mut v,
                        name.as_ptr(),
                        ylb,
                        yub,
                        reduced.linear[i],
                        ffi::SCIP_Vartype_SCIP_VARTYPE_CONTINUOUS,
                    ),
                    "SCIPcreateVarBasic(y1)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(y1)")?;
                y1_vars.push(v);
            }

            let mut pair_set = BTreeSet::<(usize, usize)>::new();
            for clique in &decomp.maximal_cliques {
                for i in 0..clique.len() {
                    for j in (i + 1)..clique.len() {
                        let a = clique[i];
                        let b = clique[j];
                        let e = if a < b { (a, b) } else { (b, a) };
                        pair_set.insert(e);
                    }
                }
            }

            let mut y2_vars = HashMap::<(usize, usize), *mut ffi::SCIP_VAR>::new();
            for (a, b) in pair_set {
                let name = CString::new(format!("y_{a}_{b}")).map_err(|e| e.to_string())?;
                let coef = *reduced.quad.get(&(a, b)).unwrap_or(&0.0);
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
                    "SCIPcreateVarBasic(y2)",
                )?;
                scip_call(ffi::SCIPaddVar(scip, v), "SCIPaddVar(y2)")?;
                y2_vars.insert((a, b), v);
            }

            let mut l_vars_all = Vec::<*mut ffi::SCIP_VAR>::new();

            for (ci, clique) in decomp.maximal_cliques.iter().enumerate() {
                let k = clique.len();
                let nmat = k + 1;
                let l_bound = (nmat as f64).sqrt().max(1.0);

                let mut l_vars = Vec::<*mut ffi::SCIP_VAR>::with_capacity(nmat * (nmat + 1) / 2);
                for i in 0..nmat {
                    for j in 0..=i {
                        let name =
                            CString::new(format!("l_{ci}_{i}_{j}")).map_err(|e| e.to_string())?;
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
                        l_vars_all.push(v);
                    }
                }

                for i in 0..nmat {
                    for j in i..nmat {
                        let mut lin_vars = Vec::<*mut ffi::SCIP_VAR>::new();
                        let mut lin_coef = Vec::<f64>::new();
                        let mut rhs = 0.0f64;

                        match (i, j) {
                            (0, 0) => {
                                rhs = -1.0;
                            }
                            (0, jj) => {
                                let v = clique[jj - 1];
                                lin_vars.push(y1_vars[v]);
                                lin_coef.push(1.0);
                            }
                            (ii, 0) => {
                                let v = clique[ii - 1];
                                lin_vars.push(y1_vars[v]);
                                lin_coef.push(1.0);
                            }
                            (ii, jj) if ii == jj => {
                                let v = clique[ii - 1];
                                match reduced.var_type {
                                    VarType::Bin => {
                                        lin_vars.push(y1_vars[v]);
                                        lin_coef.push(1.0);
                                    }
                                    VarType::Spin => {
                                        rhs = -1.0;
                                    }
                                }
                            }
                            (ii, jj) => {
                                let a = clique[ii - 1];
                                let b = clique[jj - 1];
                                let e = if a < b { (a, b) } else { (b, a) };
                                if let Some(v) = y2_vars.get(&e) {
                                    lin_vars.push(*v);
                                    lin_coef.push(1.0);
                                }
                            }
                        }

                        let mut qv1 = Vec::<*mut ffi::SCIP_VAR>::new();
                        let mut qv2 = Vec::<*mut ffi::SCIP_VAR>::new();
                        let mut qcoef = Vec::<f64>::new();
                        for r in 0..=i.min(j) {
                            qv1.push(l_vars[tri_index(i, r)]);
                            qv2.push(l_vars[tri_index(j, r)]);
                            qcoef.push(-1.0);
                        }

                        let cname =
                            CString::new(format!("psd_{ci}_{i}_{j}")).map_err(|e| e.to_string())?;
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
                                if lin_coef.is_empty() {
                                    ptr::null_mut()
                                } else {
                                    lin_coef.as_mut_ptr()
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
            }

            let timelimit_param = CString::new("limits/time").map_err(|e| e.to_string())?;
            let total_psd_dim: usize = decomp.maximal_cliques.iter().map(|c| c.len() + 1).sum();
            let tlim = (0.25 + 0.003 * total_psd_dim as f64).clamp(0.25, 2.5);
            let _ = ffi::SCIPsetRealParam(scip, timelimit_param.as_ptr(), tlim);

            scip_call(ffi::SCIPsolve(scip), "SCIPsolve")?;

            let dual = ffi::SCIPgetDualbound(scip);
            let cheap_fallback = reduced
                .linear
                .iter()
                .map(|&c| termwise_min(c, reduced.var_type))
                .sum::<f64>()
                + reduced
                    .quad
                    .values()
                    .map(|&c| termwise_min(c, reduced.var_type))
                    .sum::<f64>();

            let lb = if dual.is_finite() {
                base + dual
            } else {
                base + cheap_fallback
            };

            for v in &mut y1_vars {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(y1)")?;
            }
            for v in y2_vars.values_mut() {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(y2)")?;
            }
            for v in &mut l_vars_all {
                scip_call(ffi::SCIPreleaseVar(scip, v), "SCIPreleaseVar(L)")?;
            }

            Ok(lb)
        })();

        let free_res = scip_call(ffi::SCIPfree(&mut scip), "SCIPfree");
        if let Err(e) = free_res {
            log::debug!("SCIP free failed: {e}");
        }

        result
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_fill_cycle_adds_fill_edge() {
        let mut quad = HashMap::<(usize, usize), f64>::new();
        quad.insert((0, 1), 1.0);
        quad.insert((1, 2), 1.0);
        quad.insert((2, 3), 1.0);
        quad.insert((0, 3), 1.0);

        let (_adj, _peo, fill) = min_fill_completion(4, &quad);
        assert!(!fill.is_empty());
    }

    #[test]
    fn peo_cliques_on_path() {
        let mut quad = HashMap::<(usize, usize), f64>::new();
        quad.insert((0, 1), 1.0);
        quad.insert((1, 2), 1.0);

        let (adj, peo, _fill) = min_fill_completion(3, &quad);
        let cliques = maximal_cliques_from_peo(&adj, &peo);

        assert!(cliques.iter().any(|c| c == &vec![0, 1]));
        assert!(cliques.iter().any(|c| c == &vec![1, 2]));
        assert!(running_intersection_holds(&cliques));
    }
}
