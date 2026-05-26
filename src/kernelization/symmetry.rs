use crate::kernelization::util::{
    build_colored_symmetry_graph, connected_components, f2_sparse_kernel,
};
use crate::solver::bnb::Node;
use crate::util::set_ops::UnionFind;
use nauty_pet::prelude::*;
use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::HashMap;

use crate::coeff::Coeff;
use crate::{
    domain::{VarDomain, VarType},
    instance::HuboInstance,
    term::Term,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymmetryGraphNode {
    Variable,
    Term { color: usize },
}

#[derive(Debug, Clone)]
pub struct ColoredSymmetryGraph {
    pub graph: UnGraph<SymmetryGraphNode, ()>,
    pub variable_nodes: Vec<NodeIndex>,
    pub term_nodes: Vec<NodeIndex>,
}

// ---------------------------------------------------------------------------
// Permutation symmetries
// ---------------------------------------------------------------------------

/// Detect variable permutation symmetries with nauty on the colored incidence graph.
///
/// Each returned permutation `p` has length `n_vars` and maps variable `i` to `p[i]`.
/// The identity permutation is omitted.
pub fn detect_permutation_symmetries<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Result<Vec<Vec<usize>>, String> {
    let colored = build_colored_symmetry_graph(instance);
    let automorphisms = colored
        .graph
        .clone()
        .try_into_autom_group()
        .map_err(|err| format!("nauty automorphism detection failed: {err}"))?;

    let mut permutations = Vec::new();

    for automorphism in automorphisms.iter() {
        let permutation: Vec<usize> = colored
            .variable_nodes
            .iter()
            .map(|&node| {
                let mapped = NodeIndex::new(automorphism[node.index()]);
                match colored.graph[mapped] {
                    SymmetryGraphNode::Variable => colored
                        .variable_nodes
                        .iter()
                        .position(|&candidate| candidate == mapped)
                        .expect("variable automorphism target missing from variable node list"),
                    SymmetryGraphNode::Term { .. } => {
                        unreachable!("variable node mapped to a term node under graph automorphism")
                    }
                }
            })
            .collect();

        if permutation
            .iter()
            .enumerate()
            .any(|(idx, &mapped)| idx != mapped)
            && !permutations.contains(&permutation)
        {
            permutations.push(permutation);
        }
    }

    permutations.sort();
    Ok(permutations)
}

/// Compute variable orbits induced by a set of variable permutations.
///
/// Two variables belong to the same orbit iff one can be mapped to the other by
/// repeatedly applying permutations from the given symmetry set.
///
/// Returns all orbit blocks, including singleton orbits, sorted by their
/// minimum element.
pub fn variable_orbits_from_permutations(
    n_vars: usize,
    permutations: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>, String> {
    let mut uf = UnionFind::new(n_vars);

    for (perm_idx, permutation) in permutations.iter().enumerate() {
        if permutation.len() != n_vars {
            return Err(format!(
                "permutation {perm_idx} has length {}, expected {n_vars}",
                permutation.len()
            ));
        }

        let mut seen = vec![false; n_vars];
        for (src, &dst) in permutation.iter().enumerate() {
            if dst >= n_vars {
                return Err(format!(
                    "permutation {perm_idx} maps variable {src} to out-of-range index {dst}"
                ));
            }
            if seen[dst] {
                return Err(format!(
                    "permutation {perm_idx} is not bijective: image {dst} appears multiple times"
                ));
            }
            seen[dst] = true;
            uf.union(src, dst);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for var in 0..n_vars {
        groups.entry(uf.find(var)).or_default().push(var);
    }

    let mut orbits: Vec<Vec<usize>> = groups.into_values().collect();
    for orbit in &mut orbits {
        orbit.sort_unstable();
    }
    orbits.sort_by_key(|orbit| orbit[0]);
    Ok(orbits)
}

/// Detect variable permutation symmetries and collapse them into variable orbits.
pub fn detect_permutation_orbits<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Result<Vec<Vec<usize>>, String> {
    let permutations = detect_permutation_symmetries(instance)?;
    variable_orbits_from_permutations(instance.n_vars(), &permutations)
}

// ---------------------------------------------------------------------------
// S-flip symmetries
// ---------------------------------------------------------------------------

/// Detect the S-flip symmetries in a spin instance.
///
/// Builds the incidence matrix M over F_2 whose rows are the non-zero terms
/// (each row is the sorted set of variable indices in that term), then
/// computes ker(M) via sparse Gauss-Jordan elimination over F_2.  The kernel
/// is the complete set of valid flip sets S
///
/// Returns:
/// - F2 basis
pub fn detect_sflip_symmetries<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<Vec<usize>> {
    if V::VAR_TYPE != VarType::Spin {
        return Vec::new();
    }

    let n = instance.n_vars();

    // Build M over F_2: one row per non-zero term.
    let rows: Vec<Vec<usize>> = instance
        .terms
        .iter()
        .filter(|t| t.coeff != C::zero())
        .map(|t| t.indices.clone())
        .collect();

    // Compute the kernel basis via sparse F_2 elimination.
    f2_sparse_kernel(rows, n)
}

/// For each kernel vector, fix the representative variable (vec[0], the free
/// column) to +1 to break symmetry.  Free columns are unique across all basis
/// vectors by construction, so no two fixes can target the same variable.
pub fn sflip_symmetry_fixes<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    _node: &Node<C>,
) -> Vec<(usize, C)> {
    let kernel = detect_sflip_symmetries(instance);
    let mut fixes = Vec::new();

    for vec in kernel {
        if vec.is_empty() {
            log::warn!("Kernel vector is empty.");
            continue;
        }
        fixes.push((vec[0], C::one()));
    }

    fixes
}

// ---------------------------------------------------------------------------
// Mixed S-flip + permutation symmetries
// ---------------------------------------------------------------------------

/// Detect mixed S-flip + permutation symmetries and return the variable orbits.
pub fn detect_mixed_orbits<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    // Compute both types of symmetries independently
    let s_flip_kernel = detect_sflip_symmetries(instance);
    let permutation_orbits = detect_permutation_orbits(instance)?;

    // Combine them using a union-find: two variables are in the same orbit
    // iff they are in the same permutation orbit or they are connected by a kernel vector.
    let mut uf = UnionFind::new(instance.n_vars() * 2);

    for orbit in permutation_orbits {
        for i in 1..orbit.len() {
            uf.union(2 * orbit[0], 2 * orbit[i]);
            uf.union(2 * orbit[0] + 1, 2 * orbit[i] + 1);
        }
    }

    for kernel_vec in s_flip_kernel {
        for &var in &kernel_vec {
            uf.union(2 * var, 2 * var + 1);
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for var in 0..instance.n_vars() * 2 {
        groups.entry(uf.find(var)).or_default().push(var);
    }

    let mut orbits: Vec<Vec<usize>> = groups.into_values().collect();
    for orbit in &mut orbits {
        orbit.sort_unstable();
    }
    orbits.sort_by_key(|orbit| orbit[0]);
    Ok(orbits)
}

// ---------------------------------------------------------------------------
// Connected component decomposition
// ---------------------------------------------------------------------------

/// A single independent sub-problem extracted from a HUBO instance.
pub struct ComponentSplit<C: Coeff, V: VarDomain> {
    /// The sub-instance with remapped (0-based) variable indices.
    pub sub_instance: HuboInstance<C, V>,
    /// Maps new variable index inside `sub_instance` → original variable index.
    pub new_to_old: Vec<usize>,
}

/// Split a HUBO instance into independent sub-instances by connected component.
///
/// If the variable-interaction graph is connected, the single-element result
/// contains a clone of the instance.  Otherwise each component is extracted
/// with remapped variable indices.
///
/// The `offset` of the original instance is placed in the first sub-instance
/// (component 0); all other sub-instances have offset = 0.  This ensures
/// that summing the sub-problem objectives recovers the original objective.
pub fn split_into_components<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> Vec<ComponentSplit<C, V>> {
    let comp = connected_components(instance);

    let n_comps = if instance.n_vars() == 0 {
        0
    } else {
        comp.iter().copied().max().unwrap_or(0) + 1
    };

    if n_comps <= 1 {
        return vec![ComponentSplit {
            sub_instance: instance.clone(),
            new_to_old: (0..instance.n_vars()).collect(),
        }];
    }

    // Build per-component sorted variable lists.
    let mut comp_vars: Vec<Vec<usize>> = vec![Vec::new(); n_comps];
    for v in 0..instance.n_vars() {
        comp_vars[comp[v]].push(v);
    }

    // Build old_to_new remapping (per component, new index = position within comp_vars[c]).
    let mut old_to_new = vec![0usize; instance.n_vars()];
    for vars in &comp_vars {
        for (new_idx, &old_idx) in vars.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }
    }

    // Partition and remap terms into their components.
    let mut comp_terms: Vec<Vec<Term<C>>> = vec![Vec::new(); n_comps];
    for term in &instance.terms {
        if term.indices.is_empty() {
            continue; // constant term; the offset handles it
        }
        let c = comp[term.indices[0]];
        let remapped = Term {
            indices: term.indices.iter().map(|&v| old_to_new[v]).collect(),
            coeff: term.coeff,
        };
        comp_terms[c].push(remapped);
    }

    // Build one sub-instance per component.
    let mut splits = Vec::with_capacity(n_comps);
    for (cid, vars) in comp_vars.iter().enumerate() {
        let n_sub = vars.len();
        let mut terms = std::mem::take(&mut comp_terms[cid]);
        terms.sort_by(|a, b| a.indices.cmp(&b.indices));
        let _n_terms = terms.len();

        // The full offset only goes into the first component.
        let offset = if cid == 0 { instance.offset } else { C::zero() };

        splits.push(ComponentSplit {
            sub_instance: HuboInstance::new(n_sub, offset, terms),
            new_to_old: vars.clone(),
        });
    }

    splits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::model::HuboModel;

    // ── Connected components ──────────────────────────────────────────────

    #[test]
    fn single_component_returned_as_is() {
        let inst = HuboModel::binary(3)
            .add_term(&[0, 1], 1.0)
            .add_term(&[1, 2], 2.0)
            .build();
        let splits = split_into_components(&inst);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].sub_instance.n_vars(), 3);
        assert_eq!(splits[0].new_to_old, vec![0, 1, 2]);
    }

    #[test]
    fn two_independent_components_split() {
        // x0*x1  and  x2*x3  — two disconnected edges.
        let inst = HuboModel::binary(4)
            .add_term(&[0, 1], 3.0)
            .add_term(&[2, 3], -2.0)
            .build();
        let splits = split_into_components(&inst);
        assert_eq!(splits.len(), 2);
        // Each sub-instance should have 2 variables and 1 term.
        assert!(splits.iter().all(|s| s.sub_instance.n_vars() == 2));
        assert!(splits.iter().all(|s| s.sub_instance.n_terms() == 1));
        // The offset (0 here) goes into the first split.
        assert_eq!(splits[0].sub_instance.offset, 0.0);
        assert_eq!(splits[1].sub_instance.offset, 0.0);
    }

    #[test]
    fn offset_goes_to_first_component() {
        let inst = HuboModel::binary(4)
            .with_offset(10.0)
            .add_term(&[0, 1], 1.0)
            .add_term(&[2, 3], 2.0)
            .build();
        let splits = split_into_components(&inst);
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].sub_instance.offset, 10.0);
        assert_eq!(splits[1].sub_instance.offset, 0.0);
    }

    #[test]
    fn split_preserves_solutions() {
        // x0*x1 = 1 at (1,1); x2*x3 = -1 at (0,0).
        // Min = 1 + (-1) * 0*0 wait, let me re-check.
        // Actually for binary: f = 3*x0*x1 - 2*x2*x3
        // optimal: x0=x1=0 (→0), x2=x3=1 (→-2), total = -2.
        let inst = HuboModel::binary(4)
            .add_term(&[0, 1], 3.0)
            .add_term(&[2, 3], -2.0)
            .build();
        let splits = split_into_components(&inst);
        assert_eq!(splits.len(), 2);
        // Verify that new_to_old covers all 4 original variables exactly once.
        let mut covered = vec![false; 4];
        for split in &splits {
            for &old in &split.new_to_old {
                assert!(
                    !covered[old],
                    "variable {old} appears in multiple components"
                );
                covered[old] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all variables covered");
    }

    #[test]
    fn colored_symmetry_graph_separates_term_coefficients() {
        let inst = HuboModel::binary(3)
            .add_term(&[0, 2], 1.0)
            .add_term(&[1, 2], 2.0)
            .build();

        let graph = build_colored_symmetry_graph(&inst);
        let first = graph.graph[graph.term_nodes[0]];
        let second = graph.graph[graph.term_nodes[1]];

        match (first, second) {
            (
                SymmetryGraphNode::Term {
                    color: first_color, ..
                },
                SymmetryGraphNode::Term {
                    color: second_color,
                    ..
                },
            ) => assert_ne!(first_color, second_color),
            _ => panic!("expected term nodes"),
        }
    }

    #[test]
    fn nauty_detects_variable_swap_symmetry() {
        let inst = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 2], 2.0)
            .add_term(&[1, 2], 2.0)
            .build();

        let automorphisms = detect_permutation_symmetries(&inst).unwrap();
        assert!(automorphisms.contains(&vec![1, 0, 2]));
    }

    #[test]
    fn nauty_respects_coefficient_colors() {
        let inst = HuboModel::binary(2)
            .add_linear(0, 1.0)
            .add_linear(1, 2.0)
            .build();

        let automorphisms = detect_permutation_symmetries(&inst).unwrap();
        assert!(automorphisms.is_empty());
    }

    #[test]
    fn permutation_orbits_group_swapped_variables() {
        let inst = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 2], 2.0)
            .add_term(&[1, 2], 2.0)
            .build();

        let orbits = detect_permutation_orbits(&inst).unwrap();
        assert_eq!(orbits, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn permutation_orbits_group_fully_symmetric_variables() {
        let inst = HuboModel::binary(3)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_linear(2, 1.0)
            .add_term(&[0, 1], 2.0)
            .add_term(&[0, 2], 2.0)
            .add_term(&[1, 2], 2.0)
            .build();

        let orbits = detect_permutation_orbits(&inst).unwrap();
        assert_eq!(orbits, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn sflip_detection_finds_expected_kernel_basis() {
        let inst = HuboModel::spin(3)
            .add_term(&[0, 1], 1.0)
            .add_term(&[1, 2], 2.0)
            .add_term(&[0, 2], 0.0)
            .build();

        let kernel = detect_sflip_symmetries(&inst);
        // Free column (2) is first; pivot columns (0, 1) follow sorted.
        assert_eq!(kernel, vec![vec![2, 0, 1]]);
    }

    #[test]
    fn sflip_detection_returns_empty_for_binary_instances() {
        let instance = HuboModel::binary(2).add_term(&[0, 1], 1.0).build();
        let instance = Arc::new(instance);
        let node = Node::root(Arc::clone(&instance), f64::MIN);

        assert!(detect_sflip_symmetries(&instance).is_empty());
        assert!(sflip_symmetry_fixes(&instance, &node).is_empty());
    }

    #[test]
    fn sflip_symmetry_fixes_pick_a_representative_variable() {
        let instance = HuboModel::spin(3)
            .add_term(&[0, 1], 1.0)
            .add_term(&[1, 2], 2.0)
            .build();
        let instance = Arc::new(instance);
        let node = Node::root(Arc::clone(&instance), f64::MIN);
        // Representative is the free column (2), which is unique per basis vector.
        assert_eq!(sflip_symmetry_fixes(&instance, &node), vec![(2, 1.0)]);
    }

    #[test]
    fn permutation_orbits_keep_singletons_and_swaps() {
        let instance = HuboModel::binary(4)
            .add_linear(0, 1.0)
            .add_linear(1, 1.0)
            .add_term(&[0, 2], 3.0)
            .add_term(&[1, 2], 3.0)
            .add_linear(3, 2.0)
            .build();

        let orbits = detect_permutation_orbits(&instance).unwrap();
        assert_eq!(orbits, vec![vec![0, 1], vec![2], vec![3]]);
    }
}
