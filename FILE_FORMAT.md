# HUBO-TL File Format Specification (Term List)

## 1. Purpose

HUBO-TL encodes a **Higher-Order Unconstrained Binary Optimization** objective function as a sparse list of weighted monomials over binary variables. It supports:

* variables in **binary** domain $x_i \in {0,1}$ or **spin** domain $s_i \in {-1,+1}$,
* polynomial terms of **arbitrary degree** ($k \ge 1$),
* an optional constant **offset**,
* simple parsing and stable canonicalization for benchmarking and solver pipelines.

A HUBO-TL instance represents the objective
$$
\min_{v \in D^n}; f(v) \;=\; \text{OFFSET} + \sum_{t=1}^{M} c_t \prod_{i \in S_t} v_i
$$
where:

* $D={0,1}$ if `VAR_TYPE BIN`, and $D={-1,+1}$ if `VAR_TYPE SPIN`,
* each $S_t$ is a nonempty set (implemented as a list) of variable indices,
* $c_t \in \mathbb{R}$ is the coefficient of term $t$,
* $M$ is the number of (non-constant) terms.

## 2. File Encoding and Lexical Rules

* The file is **plain text**, recommended UTF-8.
* Lines are separated by `\n` (CRLF allowed).
* Whitespace separates tokens; multiple spaces/tabs are allowed.
* **Comments**: a `#` character starts a comment that extends to end of line. Comments are ignored.
* Blank/empty lines are ignored.

Example:

```
7 9 -2.0   # term on variables 7 and 9
```

## 3. Overall Structure

A file consists of:

1. A **required header block** (keyword lines).
2. Zero or more **metadata lines** (`META`).
3. A **term list** of exactly `M` terms (unless otherwise stated; see validation).

Header keywords may appear in any order, but each required keyword must appear exactly once.

## 4. Header Keywords (Required)

### 4.1 Magic and version

The first non-empty, non-comment line **must** be:

```
HUBO 1
```

This identifies the format and version.

### 4.2 Variable type

```
VAR_TYPE BIN
```

or

```
VAR_TYPE SPIN
```

* `BIN` means variables are interpreted as (x_i \in {0,1}).
* `SPIN` means variables are interpreted as (s_i \in {-1,+1}).

### 4.3 Number of variables

```
N <n_vars>
```

* `<n_vars>` is a positive integer.
* Variables are indexed by integers in a contiguous range determined by `INDEX_BASE` (Section 5.1).

### 4.4 Number of terms

```
M <n_terms>
```

* `<n_terms>` is a nonnegative integer.
* `M` counts the number of **non-constant** monomial terms appearing in the term list (Section 6).
* The constant term must be represented via `OFFSET` (Section 4.5), not as an empty monomial.

### 4.5 Offset (Optional but recommended)

```
OFFSET <float>
```

* `<float>` is a real number in decimal or scientific notation (e.g., `-3.25`, `1e-3`).
* If omitted, the offset is defined to be `0`.

## 5. Optional Parameters

### 5.1 Index base 

```
INDEX_BASE 0
```

or

```
INDEX_BASE 1
```

* Default is `0` if not specified.
* If `INDEX_BASE 0`, valid variable indices are integers in `[0, N-1]`.
* If `INDEX_BASE 1`, valid variable indices are integers in `[1, N]`.

**Recommendation:** Use `INDEX_BASE 0` for solver internals and programming convenience. Use `INDEX_BASE 1` to match certain benchmark conventions.

### 5.2 Metadata lines 

```
META <key>=<value>
```

* `META` lines may appear anywhere after `HUBO 1` (header or body).
* `<key>` is a nonempty string without whitespace and without `=`.
* `<value>` is the remainder of the token after `=` (no whitespace).
  If you need spaces, replace them with `_` or encode/escape externally.

Examples:

```
META source=radar_instance_07
META author=alice
META scale=1e3
META note=scaled_by_1000
```

`META` lines do not affect the mathematical meaning unless your solver explicitly chooses to interpret them.

## 6. Term List (Body)

### 6.1 Term line syntax

Each term is represented on one line as:

```
i1 i2 ... ik  <coeff>
```

where:

* `k >= 1` is the number of indices on the line excluding the last token,
* each `ij` is an integer index (subject to Section 5.1 and Section 7),
* `<coeff>` is a real number (float) coefficient.

Examples:

```
7            1.5
7 9         -2.0
1 3 5 8      0.25
```

### 6.2 Degree and semantics

* A line with one index `i c` encodes the linear term (c , v_i).
* A line with indices `i j c` encodes the quadratic term (c , v_i v_j).
* Higher-order terms are analogous.

### 6.3 Constant term

* Constant terms must be represented using `OFFSET`.
* A term line with no indices is **not permitted**.

## 7. Validity Rules

### 7.1 Index validity

Every index token must be an integer in the valid range determined by `INDEX_BASE` and `N`.

### 7.2 Duplicate indices within a term

A term line **should not** contain the same variable index more than once.

However, the mathematical reduction rule depends on the variable domain:

* **`BIN`** ($x_i \in \{0,1\}$): $x_i^k = x_i$ for all $k \ge 1$, so duplicates can simply be deduplicated (keep one copy).
* **`SPIN`** ($s_i \in \{-1,+1\}$): $s_i^{2k} = 1$ and $s_i^{2k+1} = s_i$, so duplicate indices must be removed **in pairs**.  An index appearing an even number of times vanishes entirely (contributes factor 1), while an odd count keeps exactly one copy.  If all indices cancel the term becomes a constant that is absorbed into the offset.

A reader SHOULD accept such lines by canonicalizing according to the rules above, and SHOULD emit a warning.  A writer MUST NOT emit duplicate indices.

### 7.3 Term count consistency

* The file SHOULD contain exactly `M` term lines.
* A strict parser MUST reject files with a different number of term lines than `M`.
* A permissive parser MAY accept files where the term list length differs from `M`, but must then treat `M` as advisory and should warn.

(For benchmarking reproducibility, strict parsing is recommended.)

## 8. Canonicalization (Recommended Reader Behavior)

To make instance handling robust and deterministic, a reader SHOULD canonicalize the term list into an internal representation as follows:

1. **Reduce indices within each term** according to the variable domain:
   * **`BIN`**: sort and deduplicate indices ($x^k = x$).
   * **`SPIN`**: count occurrences of each index; keep one copy for odd counts, remove entirely for even counts ($s^{2k}=1$, $s^{2k+1}=s$). If all indices cancel, add the coefficient to `OFFSET`.
2. **Sort the remaining indices** in strictly increasing order.
3. **Merge identical monomials** by summing coefficients:

   * Two terms are identical if they have the same sorted index list.
4. **Drop zero (or near-zero) coefficients**:

   * Optionally treat $|c| < \varepsilon$ as zero with a fixed tolerance (e.g., $\varepsilon=10^{-12}$).
5. Preserve `OFFSET` as-is (plus any constants absorbed from step 1).

After canonicalization, the internal term count may be smaller than the file’s `M` due to merges and eliminations.

**Important:** This format defines `M` as the number of term lines in the file (pre-canonicalization). Your solver can store both:

* `M_file` (declared), for validation,
* `M_canon` (after merges), for analysis.

## 9. Objective Direction

HUBO-TL encodes an objective function only. The default interpretation is **minimization**:
$$
\min f(v).
$$
If a maximization instance is desired, it should be converted by negating coefficients and offset before writing.

(You can optionally add `META sense=max` as advisory, but the *spec-defined* meaning is minimization.)

## 10. Example File

```
HUBO 1
VAR_TYPE BIN
INDEX_BASE 0
N 10
M 5
OFFSET -3.25
META source=radar_instance_07
META scale=1e3

7           1.5
7 9        -2.0
1 3 5 8     0.25
2 4         7
2 4         1    # duplicate monomial; should be merged to (2,4):8
```
