/// Compute the symmetric difference of two sorted arrays.
/// Input:
/// - `a`: first sorted array
/// - `b`: second sorted array
///
/// Output:
/// - a sorted vector containing the symmetric difference of `a` and `b`.
///
/// Example:
/// ```
/// use hues::util::set_ops::sym_diff;
///
/// let a = vec![1, 2, 3, 4];
/// let b = vec![3, 4, 5, 6];
/// assert_eq!(sym_diff(&a, &b), vec![1, 2, 5, 6]);
/// ```
pub fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut ai, mut bi) = (0, 0);
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => {
                result.push(a[ai]);
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[bi]);
                bi += 1;
            }
            std::cmp::Ordering::Equal => {
                ai += 1;
                bi += 1;
            }
        }
    }
    result.extend_from_slice(&a[ai..]);
    result.extend_from_slice(&b[bi..]);
    result
}

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}
