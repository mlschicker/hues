use std::collections::HashMap;

use petgraph::graph::UnGraph;

use crate::{
    Coeff,
    domain::VarDomain,
    instance::HuboInstance,
    kernelization::symmetry::{ColoredSymmetryGraph, SymmetryGraphNode},
    util::set_ops::{UnionFind, sym_diff},
};

/// Compute a basis for the kernel of a binary matrix M over F_2 via sparse
/// Gauss-Jordan elimination.
///
/// `rows`: each row is a sorted `Vec<usize>` of column indices where M = 1.
/// `n_cols`: total number of columns.
///
/// Returns a list of kernel basis vectors, each represented as a sorted
/// `Vec<usize>` of column indices where the vector equals 1.
pub fn f2_sparse_kernel(rows: Vec<Vec<usize>>, n_cols: usize) -> Vec<Vec<usize>> {
    let m = rows.len();
    let mut mat = rows;

    // col_to_pivot_row[c] = the mat-row that has column c as its pivot, if any.
    let mut col_to_pivot_row: Vec<Option<usize>> = vec![None; n_cols];
    let mut pivot_count = 0;

    for (col, pivot_row) in col_to_pivot_row.iter_mut().enumerate() {
        // Find the first row at or after pivot_count that has this column set.
        let Some(r) = (pivot_count..m).find(|&r| mat[r].binary_search(&col).is_ok()) else {
            continue;
        };

        mat.swap(r, pivot_count);
        *pivot_row = Some(pivot_count);

        // Full Gauss-Jordan: eliminate `col` from every other row.
        let pivot_row = mat[pivot_count].clone();
        for (r2, row) in mat.iter_mut().enumerate() {
            if r2 != pivot_count && row.binary_search(&col).is_ok() {
                *row = sym_diff(row, &pivot_row);
            }
        }

        pivot_count += 1;
    }

    // One kernel basis vector per free column (column without a pivot).
    // The free column is placed first (index 0) so callers can use vec[0] as a
    // guaranteed-unique representative — each basis vector has a distinct free
    // column by construction.  The remaining pivot columns are sorted.
    (0..n_cols)
        .filter(|&col| col_to_pivot_row[col].is_none())
        .map(|free_col| {
            // The basis vector for free_col has a 1 at free_col, plus a 1 at
            // each pivot column p whose pivot row still has free_col set.
            let mut pivot_cols: Vec<usize> = (0..n_cols)
                .filter(|&pivot_col| {
                    col_to_pivot_row[pivot_col]
                        .is_some_and(|row| mat[row].binary_search(&free_col).is_ok())
                })
                .collect();
            pivot_cols.sort_unstable();
            let mut v = vec![free_col];
            v.extend_from_slice(&pivot_cols);
            v
        })
        .collect()
}

/// Build the colored bipartite graph used for permutation symmetry detection.
///
/// Construction:
/// - create a variable node `v_i` for each variable `i`, all with the same color
/// - create a term node `u_S` for each term `S`, colored by its coefficient class
/// - connect `v_i` to `u_S` iff `i ∈ S`
pub fn build_colored_symmetry_graph<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
) -> ColoredSymmetryGraph {
    let mut graph = UnGraph::<SymmetryGraphNode, ()>::default();
    let mut variable_nodes = Vec::with_capacity(instance.n_vars());
    let mut term_nodes = Vec::with_capacity(instance.terms.len());
    let mut coeff_colors: Vec<C> = Vec::new();

    for _ in 0..instance.n_vars() {
        variable_nodes.push(graph.add_node(SymmetryGraphNode::Variable));
    }

    for term in &instance.terms {
        // this can be made more efficient
        let color = coeff_colors
            .iter()
            .position(|coeff| *coeff == term.coeff)
            .unwrap_or_else(|| {
                coeff_colors.push(term.coeff);
                coeff_colors.len() - 1
            });

        let term_node = graph.add_node(SymmetryGraphNode::Term { color });
        term_nodes.push(term_node);

        for &var in &term.indices {
            graph.add_edge(variable_nodes[var], term_node, ());
        }
    }

    ColoredSymmetryGraph {
        graph,
        variable_nodes,
        term_nodes,
    }
}

/// Compute the connected components of the variable-interaction graph.
///
/// Two variables are in the same component iff they co-occur in at least one
/// term.  Variables that appear in no term are singletons.
///
/// Returns a vector of length `n_vars` where `comp[v]` is the component ID
/// of variable `v`.  Component IDs are 0-based and contiguous.
pub fn connected_components<C: Coeff, V: VarDomain>(instance: &HuboInstance<C, V>) -> Vec<usize> {
    let n = instance.n_vars();
    let mut uf = UnionFind::new(n);

    for term in &instance.terms {
        if term.indices.len() >= 2 {
            let first = term.indices[0];
            for &v in &term.indices[1..] {
                uf.union(first, v);
            }
        }
    }

    // Assign contiguous component IDs in order of first appearance.
    let mut root_to_id: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    let mut comp = vec![0usize; n];
    for (idx, v) in comp.iter_mut().enumerate() {
        let root = uf.find(idx);
        let id = *root_to_id.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *v = id;
    }
    comp
}
