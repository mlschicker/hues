//! Roof duality / QPBO persistency for binary and spin QUBO instances.
//!
//! Implements the symmetric QPBO graph construction of Kolmogorov & Zabih
//! (2004) with strong persistency deductions from Boros & Hammer (2002).
//!
//! ## Node layout for n variables
//!
//! ```text
//!   s        = 0
//!   primal i = 1 + i          (represents x_i = 1)
//!   compl  i = 1 + n + i      (represents x_i = 0)
//!   t        = 2n + 1
//! ```
//!
//! Every arc (u → v, c) is paired with its conjugate (comp(v) → comp(u), c),
//! so the minimum s-t cut is always consistent: for each variable exactly one
//! of {primal, compl} is on the source side.
//!
//! ## Deductions after max-flow
//!
//! - `primal(i)` reachable from `s` → `x_i = 1` is strongly persistent
//! - `compl(i)`  reachable from `s` → `x_i = 0` is strongly persistent
//!
//! ## Spin support
//!
//! Spin instances are binarized via `s_i = 2x_i − 1` before graph
//! construction. Only degree ≤ 2 spin instances are supported; higher
//! degrees would produce higher-order binary residuals that are unsound to
//! drop, so they return no fixings.

use std::collections::{HashMap, VecDeque};

use crate::coeff::Coeff;
use crate::domain::{VarDomain, VarType};
use crate::instance::HuboInstance;
use crate::solver::bnb::Node;

pub fn roof_dual_fixes<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    _node: &Node<C>,
) -> Vec<(usize, C)> {
    match V::VAR_TYPE {
        VarType::Bin => binary_roof_duality(instance),
        VarType::Spin => spin_roof_duality(instance),
    }
}

// ---------------------------------------------------------------------------
// Dinic's max-flow
// ---------------------------------------------------------------------------

struct Dinic {
    n: usize,
    head: Vec<usize>,
    to: Vec<usize>,
    cap: Vec<f64>,
    next: Vec<usize>,
}

const NONE: usize = usize::MAX;
const EPS: f64 = 1e-9;

impl Dinic {
    fn new(n: usize) -> Self {
        Self {
            n,
            head: vec![NONE; n],
            to: Vec::new(),
            cap: Vec::new(),
            next: Vec::new(),
        }
    }

    fn add_edge(&mut self, u: usize, v: usize, c: f64) {
        self.to.push(v);
        self.cap.push(c);
        self.next.push(self.head[u]);
        self.head[u] = self.to.len() - 1;
        // backward arc (zero initial capacity)
        self.to.push(u);
        self.cap.push(0.0);
        self.next.push(self.head[v]);
        self.head[v] = self.to.len() - 1;
    }

    fn bfs(&self, s: usize, t: usize, level: &mut [i32]) -> bool {
        level.iter_mut().for_each(|l| *l = -1);
        level[s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            let mut eid = self.head[u];
            while eid != NONE {
                let v = self.to[eid];
                if self.cap[eid] > EPS && level[v] < 0 {
                    level[v] = level[u] + 1;
                    q.push_back(v);
                }
                eid = self.next[eid];
            }
        }
        level[t] >= 0
    }

    fn dfs(
        &mut self,
        u: usize,
        t: usize,
        pushed: f64,
        level: &[i32],
        iter: &mut Vec<usize>,
    ) -> f64 {
        if u == t {
            return pushed;
        }
        while iter[u] != NONE {
            let eid = iter[u];
            let v = self.to[eid];
            let c = self.cap[eid];
            if c > EPS && level[v] == level[u] + 1 {
                let d = self.dfs(v, t, pushed.min(c), level, iter);
                if d > EPS {
                    self.cap[eid] -= d;
                    self.cap[eid ^ 1] += d;
                    return d;
                }
            }
            iter[u] = self.next[eid];
        }
        0.0
    }

    fn max_flow(&mut self, s: usize, t: usize) {
        let mut level = vec![-1i32; self.n];
        while self.bfs(s, t, &mut level) {
            let mut iter = self.head.clone();
            loop {
                let f = self.dfs(s, t, f64::INFINITY, &level, &mut iter);
                if f <= EPS {
                    break;
                }
            }
        }
    }

    fn reachable_from(&self, start: usize) -> Vec<bool> {
        let mut vis = vec![false; self.n];
        vis[start] = true;
        let mut q = VecDeque::new();
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            let mut eid = self.head[u];
            while eid != NONE {
                let v = self.to[eid];
                if self.cap[eid] > EPS && !vis[v] {
                    vis[v] = true;
                    q.push_back(v);
                }
                eid = self.next[eid];
            }
        }
        vis
    }
}

// ---------------------------------------------------------------------------
// Symmetric QPBO graph construction
// ---------------------------------------------------------------------------

/// Run symmetric QPBO on a binary QUBO and return per-variable fixings.
///
/// Returns a vector of length `n`:
/// - `Some(true)`  → `x_i = 1` is strongly persistent
/// - `Some(false)` → `x_i = 0` is strongly persistent
/// - `None`        → not determined
fn qpbo_fixings(
    n: usize,
    linear: &[f64],
    quadratic: &HashMap<(usize, usize), f64>,
) -> Vec<Option<bool>> {
    let s = 0usize;
    let t = 2 * n + 1;
    let num_nodes = 2 * n + 2;

    let pri = |i: usize| 1 + i;
    let cmp = |i: usize| 1 + n + i;

    // comp(s)=t, comp(t)=s, comp(primal i)=compl i, comp(compl i)=primal i.
    let comp = |node: usize| -> usize {
        if node == s {
            t
        } else if node == t {
            s
        } else if node <= n {
            node + n // primal i (1..=n) → compl i
        } else {
            node - n // compl i (n+1..=2n) → primal i
        }
    };

    let mut g = Dinic::new(num_nodes);

    // add_sym: arc (u→v, c) and its conjugate (comp(v)→comp(u), c).
    macro_rules! add_sym {
        ($u:expr, $v:expr, $c:expr) => {{
            let u: usize = $u;
            let v: usize = $v;
            let c: f64 = $c;
            if c > EPS {
                g.add_edge(u, v, c);
                g.add_edge(comp(v), comp(u), c);
            }
        }};
    }

    // Linear terms.
    for (i, &ai) in linear.iter().enumerate() {
        if ai > EPS {
            // Cost ai when x_i=1 (primal on S-side): arc primal→t.
            add_sym!(pri(i), t, ai);
        } else if ai < -EPS {
            // Cost |ai| when x_i=0 (compl on S-side): arc s→primal.
            add_sym!(s, pri(i), -ai);
        }
    }

    // Quadratic terms (sorted for deterministic arc order).
    let mut quad_sorted: Vec<_> = quadratic.iter().collect();
    quad_sorted.sort_unstable_by_key(|&(&(i, j), _)| (i, j));
    for (&(i, j), &bij) in quad_sorted {
        if bij > EPS {
            // Supermodular: penalise x_i=1, x_j=1.
            add_sym!(pri(i), cmp(j), bij);
        } else if bij < -EPS {
            let c = -bij;
            // Submodular: reparameterise by complementing x_i.
            // b·x_i·x_j = b·x_j + c·(1−x_i)·x_j  (c = −b > 0).
            // Linear term b·x_j (b<0) → penalises x_j=0 by c.
            // Quadratic c·(1−x_i)·x_j is supermodular in (1−x_i, x_j).
            add_sym!(s, pri(j), c); // encodes b·x_j
            add_sym!(cmp(i), cmp(j), c); // supermodular c·(1−x_i)·x_j
        }
    }

    g.max_flow(s, t);

    let reach = g.reachable_from(s);

    (0..n)
        .map(|i| {
            if reach[pri(i)] {
                Some(true) // x_i = 1
            } else if reach[cmp(i)] {
                Some(false) // x_i = 0
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Roof duality (QPBO) for binary HUBO (`x_i ∈ {0,1}`).
///
/// Projects the binary objective to a QUBO (degree ≤ 2 only) and applies
/// the symmetric s-t min-cut to derive strongly persistent variable fixings.
/// Returns empty if the maximum term degree exceeds 2.
pub fn binary_roof_duality<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<(usize, C)> {
    if V::VAR_TYPE != VarType::Bin {
        return Vec::new();
    }

    let n = instance.n_vars();
    if n == 0 {
        return Vec::new();
    }

    let max_degree = instance
        .terms
        .iter()
        .map(|t| t.indices.len())
        .max()
        .unwrap_or(0);
    if max_degree > 2 {
        return Vec::new();
    }

    let mut linear = vec![0.0f64; n];
    let mut quadratic: HashMap<(usize, usize), f64> = HashMap::new();

    for term in &instance.terms {
        let c = term.coeff.to_f64();
        match term.indices.as_slice() {
            [] => {}
            [i] => linear[*i] += c,
            [i, j] => {
                let key = if i <= j { (*i, *j) } else { (*j, *i) };
                *quadratic.entry(key).or_insert(0.0) += c;
            }
            _ => {}
        }
    }

    let fixings = qpbo_fixings(n, &linear, &quadratic);

    fixings
        .into_iter()
        .enumerate()
        .filter_map(|(i, fix)| match fix {
            Some(true) => Some((i, C::one())),
            Some(false) => Some((i, C::zero())),
            None => None,
        })
        .collect()
}

/// Roof duality (QPBO) for spin HUBO (`s_i ∈ {−1,+1}`).
///
/// Binarizes the spin objective via `s_i = 2x_i − 1`, applies the symmetric
/// s-t min-cut on the resulting binary QUBO, then maps fixings back to the
/// spin domain (`x_i = 0 → s_i = −1`, `x_i = 1 → s_i = +1`).
/// Returns empty if the maximum spin-term degree exceeds 2.
pub fn spin_roof_duality<C: Coeff, V: VarDomain>(instance: &HuboInstance<C, V>) -> Vec<(usize, C)> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let n = instance.n_vars();
    if n == 0 {
        return Vec::new();
    }

    // For degree-k spin terms (k ≥ 3), substituting s_i = 2x_i−1 produces
    // binary terms of degree k. Dropping them is unsound, so bail out.
    let max_degree = instance
        .terms
        .iter()
        .map(|t| t.indices.len())
        .max()
        .unwrap_or(0);
    if max_degree > 2 {
        return Vec::new();
    }

    // Binarize: s_i = 2x_i − 1.
    //
    // For a degree-k spin term c * ∏_{i∈S} s_i (k ≤ 2):
    //   let sign = (−1)^k
    //   linear coeff per variable i ∈ S:   −sign * 2c
    //   quadratic coeff per pair (i,j) ⊂ S: sign * 4c
    //
    // Degree 1 (k=1, sign=−1): linear[i] += 2c
    // Degree 2 (k=2, sign=+1): linear[i] += −2c, linear[j] += −2c, quad[(i,j)] += 4c
    let mut linear = vec![0.0f64; n];
    let mut quadratic: HashMap<(usize, usize), f64> = HashMap::new();

    for term in &instance.terms {
        let c = term.coeff.to_f64();
        let k = term.indices.len();
        if k == 0 {
            continue;
        }
        let sign: f64 = if k % 2 == 0 { 1.0 } else { -1.0 };
        let lin_coeff = -sign * 2.0 * c;
        let quad_coeff = sign * 4.0 * c;

        for &i in &term.indices {
            linear[i] += lin_coeff;
        }
        if k == 2 {
            let (a, b) = if term.indices[0] <= term.indices[1] {
                (term.indices[0], term.indices[1])
            } else {
                (term.indices[1], term.indices[0])
            };
            *quadratic.entry((a, b)).or_insert(0.0) += quad_coeff;
        }
    }

    let fixings = qpbo_fixings(n, &linear, &quadratic);

    // x_i = 0 → s_i = −1;  x_i = 1 → s_i = +1.
    fixings
        .into_iter()
        .enumerate()
        .filter_map(|(i, fix)| match fix {
            Some(true) => Some((i, C::one())),
            Some(false) => Some((i, -C::one())),
            None => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HuboModel;
    use std::collections::HashMap;

    fn bin_fixes(
        instance: &crate::instance::HuboInstance<f64, crate::domain::Bin>,
    ) -> HashMap<usize, f64> {
        binary_roof_duality(instance).into_iter().collect()
    }

    fn spin_fixes(
        instance: &crate::instance::HuboInstance<f64, crate::domain::Spin>,
    ) -> HashMap<usize, f64> {
        spin_roof_duality(instance).into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // Binary tests
    // -----------------------------------------------------------------------

    #[test]
    fn binary_submodular_fixes_both_to_one() {
        // f = x0 + x1 − 3·x0·x1.  Min −1 at (1,1).
        let inst = HuboModel::binary(2)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 1], -3.0)
            .build();
        let f = bin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&1.0));
        assert_eq!(f.get(&1), Some(&1.0));
    }

    #[test]
    fn binary_supermodular_ambiguous_no_fix() {
        // f = −x0 − x1 + 3·x0·x1.  Min −1 at (1,0) and (0,1) → neither persistent.
        let inst = HuboModel::binary(2)
            .add_linear(0, -1.0)
            .add_linear(1, -1.0)
            .add_term(&[0, 1], 3.0)
            .build();
        assert!(bin_fixes(&inst).is_empty());
    }

    #[test]
    fn binary_linear_only_fixes_to_zero() {
        // f = 5·x0.  Min 0 at x0=0.
        let inst = HuboModel::binary(1).add_linear(0, 5.0).build();
        let f = bin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&0.0));
    }

    #[test]
    fn binary_linear_only_fixes_to_one() {
        // f = −5·x0.  Min −5 at x0=1.
        let inst = HuboModel::binary(1).add_linear(0, -5.0).build();
        let f = bin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&1.0));
    }

    #[test]
    fn binary_empty_instance_no_fix() {
        let inst: crate::instance::HuboInstance<f64, crate::domain::Bin> =
            HuboModel::binary(0).build();
        assert!(binary_roof_duality(&inst).is_empty());
    }

    #[test]
    fn binary_degree3_skipped() {
        let inst = HuboModel::binary(3).add_term(&[0, 1, 2], -5.0).build();
        assert!(binary_roof_duality(&inst).is_empty());
    }

    #[test]
    fn binary_mixed_supermodular_unique_optimum() {
        // f = x0 + x1 − 150·x2 + 100·x0·x2 + 100·x1·x2 − 3·x0·x1
        // Unique min −150 at (0,0,1).
        let inst = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_linear(2, -150.0)
            .add_term(&[0, 1], -3.0)
            .add_term(&[0, 2], 100.0)
            .add_term(&[1, 2], 100.0)
            .build();
        let f = bin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&0.0));
        assert_eq!(f.get(&1), Some(&0.0));
        assert_eq!(f.get(&2), Some(&1.0));
    }

    // -----------------------------------------------------------------------
    // Spin tests
    // -----------------------------------------------------------------------

    #[test]
    fn spin_linear_fixes_to_minus_one() {
        // f = 5·s0.  Min at s0=−1.
        let inst = HuboModel::spin(1).add_linear(0, 5.0).build();
        let f = spin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&-1.0));
    }

    #[test]
    fn spin_linear_fixes_to_plus_one() {
        // f = −5·s0.  Min at s0=+1.
        let inst = HuboModel::spin(1).add_linear(0, -5.0).build();
        let f = spin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&1.0));
    }

    #[test]
    fn spin_ferromagnetic_fixes_both_negative() {
        // f = s0 + s1 − 3·s0·s1.
        // Values: (−1,−1)=−5, (−1,+1)=3, (+1,−1)=3, (+1,+1)=−1.
        // Unique min at (−1,−1).
        let inst = HuboModel::spin(2)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 1], -3.0)
            .build();
        let f = spin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&-1.0));
        assert_eq!(f.get(&1), Some(&-1.0));
    }

    #[test]
    fn spin_antiferromagnetic_ambiguous_no_fix() {
        // f = −s0 − s1 + 3·s0·s1.
        // Values: (−1,−1)=1, (−1,+1)=−1, (+1,−1)=−1, (+1,+1)=1.
        // Two optima: neither variable is persistent.
        let inst = HuboModel::spin(2)
            .add_linear(0, -1.0)
            .add_linear(1, -1.0)
            .add_term(&[0, 1], 3.0)
            .build();
        assert!(spin_fixes(&inst).is_empty());
    }

    #[test]
    fn spin_ferromagnetic_coupling_no_linear() {
        // f = −3·s0·s1.  Min −3 at s0=s1=±1 (two optima) → no fixings.
        let inst = HuboModel::spin(2).add_term(&[0, 1], -3.0).build();
        assert!(spin_fixes(&inst).is_empty());
    }

    #[test]
    fn spin_empty_instance_no_fix() {
        let inst: crate::instance::HuboInstance<f64, crate::domain::Spin> =
            HuboModel::spin(0).build();
        assert!(spin_roof_duality(&inst).is_empty());
    }

    #[test]
    fn spin_degree3_skipped() {
        let inst = HuboModel::spin(3).add_term(&[0, 1, 2], -5.0).build();
        assert!(spin_roof_duality(&inst).is_empty());
    }

    #[test]
    fn spin_mixed_coupling_unique_optimum() {
        // f = s0 − 10·s0·s1.
        // Values: (−1,−1) = −1−10 = −11, (−1,+1) = −1+10 = 9,
        //         (+1,−1) = 1+10 = 11,   (+1,+1) = 1−10 = −9.
        // Unique min at (−1,−1).
        let inst = HuboModel::spin(2)
            .add_linear(0, 1.0)
            .add_term(&[0, 1], -10.0)
            .build();
        let f = spin_fixes(&inst);
        assert_eq!(f.get(&0), Some(&-1.0));
        assert_eq!(f.get(&1), Some(&-1.0));
    }
}
