use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::coeff::Coeff;
use crate::error::{ParseError, ParseWarning};
use crate::{
    domain::{Bin, Spin, VarDomain, VarType},
    instance::{HuboInstance, HuboInstanceEnum},
    term::Term,
};

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Strip comments and trim whitespace. Returns `None` for blank lines.
fn strip_line(line: &str) -> Option<&str> {
    let s = match line.find('#') {
        Some(pos) => &line[..pos],
        None => line,
    };
    let s = s.trim();
    if s.is_empty() { None } else { Some(s) }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a HUBO-TL formatted string into a [`HuboInstanceEnum`].
pub fn parse<C: Coeff>(
    input: &str,
) -> Result<(HuboInstanceEnum<C>, Vec<(String, String)>), ParseError> {
    let lines: Vec<(usize, &str)> = input
        .lines()
        .enumerate()
        .filter_map(|(i, l)| strip_line(l).map(|s| (i + 1, s)))
        .collect();

    if lines.is_empty() {
        return Err(ParseError::EmptyFile);
    }

    // ---- magic line --------------------------------------------------------
    let (_, first) = lines[0];
    if first != "HUBO 1" && first != "HUSO 1" {
        return Err(ParseError::InvalidMagic(first.to_string()));
    }

    // ---- scan header & meta lines -----------------------------------------
    let mut var_type: Option<VarType> = None;
    let mut n_vars: Option<usize> = None;
    let mut n_terms: Option<usize> = None;
    let mut offset: C = C::zero();
    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut warnings: Vec<ParseWarning> = Vec::new();

    let mut seen_var_type = false;
    let mut seen_n = false;
    let mut seen_m = false;
    let mut seen_offset = false;

    let mut body_start: Option<usize> = None;

    for (idx, &(lineno, content)) in lines.iter().enumerate().skip(1) {
        let tokens: Vec<&str> = content.split_whitespace().collect();
        debug_assert!(!tokens.is_empty());

        match tokens[0] {
            "VAR_TYPE" => {
                if seen_var_type {
                    return Err(ParseError::DuplicateHeader {
                        keyword: "VAR_TYPE".into(),
                        line: lineno,
                    });
                }
                seen_var_type = true;
                if tokens.len() != 2 {
                    return Err(ParseError::InvalidHeaderValue {
                        keyword: "VAR_TYPE".into(),
                        value: tokens[1..].join(" "),
                        line: lineno,
                    });
                }
                var_type = Some(match tokens[1] {
                    "BIN" => VarType::Bin,
                    "SPIN" => VarType::Spin,
                    other => {
                        return Err(ParseError::InvalidHeaderValue {
                            keyword: "VAR_TYPE".into(),
                            value: other.into(),
                            line: lineno,
                        });
                    }
                });
            }
            "N" => {
                if seen_n {
                    return Err(ParseError::DuplicateHeader {
                        keyword: "N".into(),
                        line: lineno,
                    });
                }
                seen_n = true;
                if tokens.len() != 2 {
                    return Err(ParseError::InvalidHeaderValue {
                        keyword: "N".into(),
                        value: tokens[1..].join(" "),
                        line: lineno,
                    });
                }
                n_vars = Some(tokens[1].parse::<usize>().map_err(|_| {
                    ParseError::InvalidHeaderValue {
                        keyword: "N".into(),
                        value: tokens[1].into(),
                        line: lineno,
                    }
                })?);
            }
            "M" => {
                if seen_m {
                    return Err(ParseError::DuplicateHeader {
                        keyword: "M".into(),
                        line: lineno,
                    });
                }
                seen_m = true;
                if tokens.len() != 2 {
                    return Err(ParseError::InvalidHeaderValue {
                        keyword: "M".into(),
                        value: tokens[1..].join(" "),
                        line: lineno,
                    });
                }
                n_terms = Some(tokens[1].parse::<usize>().map_err(|_| {
                    ParseError::InvalidHeaderValue {
                        keyword: "M".into(),
                        value: tokens[1].into(),
                        line: lineno,
                    }
                })?);
            }
            "OFFSET" => {
                if seen_offset {
                    return Err(ParseError::DuplicateHeader {
                        keyword: "OFFSET".into(),
                        line: lineno,
                    });
                }
                seen_offset = true;
                if tokens.len() != 2 {
                    return Err(ParseError::InvalidHeaderValue {
                        keyword: "OFFSET".into(),
                        value: tokens[1..].join(" "),
                        line: lineno,
                    });
                }
                offset = C::parse_str(tokens[1]).map_err(|_| ParseError::InvalidHeaderValue {
                    keyword: "OFFSET".into(),
                    value: tokens[1].into(),
                    line: lineno,
                })?;
            }
            "META" => {
                let payload = content
                    .strip_prefix("META")
                    .expect("matched META prefix")
                    .trim_start();

                if payload.is_empty() {
                    return Err(ParseError::InvalidMeta {
                        line: lineno,
                        detail: "expected `META key=value`".into(),
                    });
                }

                let eq_pos = payload.find('=').ok_or_else(|| ParseError::InvalidMeta {
                    line: lineno,
                    detail: "missing `=` in META value".into(),
                })?;

                let key = payload[..eq_pos].trim();
                let value = payload[eq_pos + 1..].trim();

                if key.is_empty() {
                    return Err(ParseError::InvalidMeta {
                        line: lineno,
                        detail: "empty key".into(),
                    });
                }

                if key.contains('=') {
                    return Err(ParseError::InvalidMeta {
                        line: lineno,
                        detail: "key must not contain `=`".into(),
                    });
                }

                if key.chars().any(char::is_whitespace) {
                    return Err(ParseError::InvalidMeta {
                        line: lineno,
                        detail: "key must not contain whitespace".into(),
                    });
                }

                metadata.push((key.to_string(), value.to_string()));
            }
            "HUBO" | "HUSO" => {
                return Err(ParseError::DuplicateHeader {
                    keyword: tokens[0].into(),
                    line: lineno,
                });
            }
            "INDEX_BASE" => {
                // Auto-detected from term data; ignore the declared value.
            }
            _ => {
                body_start = Some(idx);
                break;
            }
        }
    }

    let var_type = var_type.ok_or(ParseError::MissingHeader("VAR_TYPE"))?;
    let n_vars = n_vars.ok_or(ParseError::MissingHeader("N"))?;
    let n_terms = n_terms.ok_or(ParseError::MissingHeader("M"))?;

    let body_lines: &[(usize, &str)] = match body_start {
        Some(start) => &lines[start..],
        None => &[],
    };

    let mut raw_entries: Vec<RawEntry<C>> = Vec::with_capacity(n_terms);
    let mut lo_seen = u64::MAX;
    let mut hi_seen = 0u64;

    for &(lineno, content) in body_lines {
        let tokens: Vec<&str> = content.split_whitespace().collect();

        if tokens[0] == "META" {
            return Err(ParseError::MetaInBody { line: lineno });
        }

        if tokens.len() < 2 {
            return Err(ParseError::InsufficientTokens { line: lineno });
        }

        let coeff_str = tokens[tokens.len() - 1];
        let coeff: C = C::parse_str(coeff_str).map_err(|_| ParseError::InvalidCoeff {
            token: coeff_str.into(),
            line: lineno,
        })?;

        let index_tokens = &tokens[..tokens.len() - 1];

        if index_tokens.is_empty() {
            return Err(ParseError::EmptyTerm { line: lineno });
        }

        let mut raw_indices: Vec<u64> = Vec::with_capacity(index_tokens.len());

        for &tok in index_tokens {
            let raw: u64 = tok.parse().map_err(|_| ParseError::InvalidIdx {
                token: tok.into(),
                line: lineno,
            })?;

            if raw > n_vars as u64 {
                return Err(ParseError::IndexOutOfRange {
                    index: raw,
                    line: lineno,
                });
            }

            lo_seen = lo_seen.min(raw);
            hi_seen = hi_seen.max(raw);
            raw_indices.push(raw);
        }

        raw_entries.push(RawEntry {
            lineno,
            raw_indices,
            coeff,
        });
    }

    if raw_entries.len() != n_terms {
        return Err(ParseError::TermCountMismatch {
            declared: n_terms,
            actual: raw_entries.len(),
        });
    }

    let index_offset: u64 = if lo_seen > hi_seen {
        0
    } else if lo_seen == 0 {
        0
    } else if hi_seen == n_vars as u64 {
        1
    } else {
        warnings.push(ParseWarning::AmbiguousIndexBase);
        0
    };

    // Dispatch to the typed reducer.
    let (terms, offset) = match var_type {
        VarType::Bin => reduce_terms::<C, Bin>(raw_entries, index_offset, offset, &mut warnings)?,
        VarType::Spin => reduce_terms::<C, Spin>(raw_entries, index_offset, offset, &mut warnings)?,
    };

    for warning in &warnings {
        ::log::warn!("{warning}");
    }

    let enum_inst = match var_type {
        VarType::Bin => HuboInstanceEnum::Bin(HuboInstance::new(n_vars, offset, terms)),
        VarType::Spin => HuboInstanceEnum::Spin(HuboInstance::new(n_vars, offset, terms)),
    };

    Ok((enum_inst, metadata))
}

struct RawEntry<C> {
    lineno: usize,
    raw_indices: Vec<u64>,
    coeff: C,
}

fn reduce_terms<C: Coeff, V: VarDomain>(
    raw_entries: Vec<RawEntry<C>>,
    index_offset: u64,
    mut offset: C,
    warnings: &mut Vec<ParseWarning>,
) -> Result<(Vec<Term<C>>, C), ParseError> {
    let mut terms: Vec<Term<C>> = Vec::with_capacity(raw_entries.len());

    for entry in raw_entries {
        let RawEntry {
            lineno,
            raw_indices,
            coeff,
        } = entry;

        let mut has_duplicates = false;
        {
            let mut seen_set = HashSet::with_capacity(raw_indices.len());
            for &raw in &raw_indices {
                if !seen_set.insert(raw) {
                    has_duplicates = true;
                    warnings.push(ParseWarning::DuplicateIndex {
                        index: raw,
                        line: lineno,
                    });
                }
            }
        }

        let raw_zero_based: Vec<usize> = raw_indices
            .into_iter()
            .map(|r| (r - index_offset) as usize)
            .collect();

        let mut indices = if has_duplicates {
            V::reduce_indices(&raw_zero_based)
        } else {
            let mut v = raw_zero_based;
            v.sort_unstable();
            v
        };

        indices.sort_unstable();

        if indices.is_empty() {
            offset += coeff;
            continue;
        }

        terms.push(Term { indices, coeff });
    }

    Ok((terms, offset))
}

fn parse_json_key_indices(key: &str) -> Result<Vec<usize>, ParseError> {
    let trimmed = key.trim();
    let inner = trimmed
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| {
            ParseError::InvalidJson(format!(
                "term key `{key}` must be parenthesized like `(1, 2, 3)`"
            ))
        })?
        .trim();

    if inner.is_empty() {
        return Err(ParseError::InvalidJson(format!(
            "term key `{key}` must contain at least one index"
        )));
    }

    inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                ParseError::InvalidJson(format!("term key `{key}` contains invalid index `{part}`"))
            })
        })
        .collect()
}

/// Parse a structured JSON instance produced by `HuboInstance::to_json`.
///
/// Expected shape:
/// ```json
/// { "var_type": "BIN", "n_vars": 4, "offset": 0, "terms": [{"indices":[0,1],"coeff":3}] }
/// ```
pub fn parse_json_structured<C: Coeff>(
    input: &str,
) -> Result<(HuboInstanceEnum<C>, Vec<(String, String)>), ParseError> {
    let root: Value =
        serde_json::from_str(input).map_err(|e| ParseError::InvalidJson(e.to_string()))?;
    let obj = root
        .as_object()
        .ok_or_else(|| ParseError::InvalidJson("expected a JSON object at the top level".into()))?;

    let var_type = match obj.get("var_type").and_then(|v| v.as_str()) {
        Some("BIN") => VarType::Bin,
        Some("SPIN") => VarType::Spin,
        Some(other) => {
            return Err(ParseError::InvalidJson(format!(
                "unknown var_type \"{other}\""
            )));
        }
        None => return Err(ParseError::InvalidJson("missing \"var_type\"".into())),
    };

    let n_vars = obj
        .get("n_vars")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ParseError::InvalidJson("missing or invalid \"n_vars\"".into()))?
        as usize;

    let offset_f = obj.get("offset").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let offset = C::parse_str(&offset_f.to_string())
        .map_err(|_| ParseError::InvalidJson("invalid \"offset\"".into()))?;

    let terms_arr = obj
        .get("terms")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ParseError::InvalidJson("missing \"terms\" array".into()))?;

    let mut terms = Vec::with_capacity(terms_arr.len());
    for (i, t) in terms_arr.iter().enumerate() {
        let t_obj = t
            .as_object()
            .ok_or_else(|| ParseError::InvalidJson(format!("term {i} is not an object")))?;

        let indices: Vec<usize> = t_obj
            .get("indices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ParseError::InvalidJson(format!("term {i} missing \"indices\"")))?
            .iter()
            .enumerate()
            .map(|(j, v)| {
                v.as_u64()
                    .ok_or_else(|| {
                        ParseError::InvalidJson(format!(
                            "term {i} index {j} is not a non-negative integer"
                        ))
                    })
                    .map(|u| u as usize)
            })
            .collect::<Result<_, _>>()?;

        let coeff_f = t_obj.get("coeff").and_then(|v| v.as_f64()).ok_or_else(|| {
            ParseError::InvalidJson(format!("term {i} missing or invalid \"coeff\""))
        })?;
        let coeff = C::parse_str(&coeff_f.to_string())
            .map_err(|_| ParseError::InvalidJson(format!("term {i} has unparseable coeff")))?;

        terms.push(Term { indices, coeff });
    }

    let metadata: Vec<(String, String)> = obj
        .get("metadata")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                .collect()
        })
        .unwrap_or_default();

    let inst = match var_type {
        VarType::Bin => HuboInstanceEnum::Bin(HuboInstance::new(n_vars, offset, terms)),
        VarType::Spin => HuboInstanceEnum::Spin(HuboInstance::new(n_vars, offset, terms)),
    };

    Ok((inst, metadata))
}

/// Parse a flat JSON object instance of the form `{ "(1, 2)": 3.0, "(4,)": -1.0 }`.
///
/// Defaults to SPIN domain.
pub fn parse_json<C: Coeff>(
    input: &str,
) -> Result<(HuboInstanceEnum<C>, Vec<(String, String)>), ParseError> {
    let root: Value =
        serde_json::from_str(input).map_err(|e| ParseError::InvalidJson(e.to_string()))?;
    let object = root
        .as_object()
        .ok_or_else(|| ParseError::InvalidJson("expected a JSON object at the top level".into()))?;

    let mut terms = Vec::with_capacity(object.len());
    let mut max_idx = None::<usize>;

    for (key, value) in object {
        let coeff_str = match value {
            Value::Number(n) => n.to_string(),
            _ => {
                return Err(ParseError::InvalidJson(format!(
                    "term `{key}` must map to a numeric coefficient"
                )));
            }
        };
        let coeff = C::parse_str(&coeff_str).map_err(|_| {
            ParseError::InvalidJson(format!(
                "term `{key}` has coefficient `{coeff_str}` that cannot be parsed"
            ))
        })?;

        let mut indices = parse_json_key_indices(key)?;
        indices.sort_unstable();

        for &idx in &indices {
            max_idx = Some(max_idx.map_or(idx, |cur| cur.max(idx)));
        }

        terms.push(Term { indices, coeff });
    }

    let n_vars = max_idx.map_or(0, |idx| idx + 1);

    Ok((
        HuboInstanceEnum::Spin(HuboInstance::new(n_vars, C::zero(), terms)),
        Vec::new(),
    ))
}

/// Parse either HUBO/HUSO text, the structured JSON format, or the legacy
/// tuple-key JSON format.  Format is auto-detected from content and extension.
pub fn parse_auto<C: Coeff>(
    input: &str,
    path: Option<&str>,
) -> Result<(HuboInstanceEnum<C>, Vec<(String, String)>), ParseError> {
    let trimmed = input.trim_start();
    let looks_like_json = trimmed.starts_with('{') || path.is_some_and(|p| p.ends_with(".json"));

    if looks_like_json {
        let root: Value =
            serde_json::from_str(input).map_err(|e| ParseError::InvalidJson(e.to_string()))?;
        if root.get("var_type").is_some() {
            parse_json_structured(input)
        } else {
            parse_json(input)
        }
    } else {
        parse(input)
    }
}

// ---------------------------------------------------------------------------
// Solution file parser
// ---------------------------------------------------------------------------

/// Parse a HUES solution file and return variable values in the original
/// domain (0/1 for BIN, -1/+1 for SPIN).
pub fn parse_solution_file<C: Coeff>(
    path: &str,
    n_vars: usize,
    var_type: VarType,
) -> Result<Vec<C>, String> {
    match var_type {
        VarType::Bin => parse_solution_file_typed::<C, Bin>(path, n_vars),
        VarType::Spin => parse_solution_file_typed::<C, Spin>(path, n_vars),
    }
}

pub fn parse_solution_file_typed<C: Coeff, V: VarDomain>(
    path: &str,
    n_vars: usize,
) -> Result<Vec<C>, String> {
    let contents = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;

    let mut in_solution = false;
    let mut vals: HashMap<usize, C> = HashMap::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed == "SOLUTION" {
            in_solution = true;
            continue;
        }
        if !in_solution {
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.splitn(3, '=').collect();
        if parts.len() != 2 {
            return Err(format!("malformed solution line: {trimmed}"));
        }
        let label = parts[0].trim();
        let value: C = C::parse_str(parts[1].trim())
            .map_err(|_| format!("invalid value in solution line: {trimmed}"))?;

        let idx_str = if label.starts_with('x') || label.starts_with('s') {
            &label[1..]
        } else {
            return Err(format!("unexpected variable label: {label}"));
        };
        let idx: usize = idx_str
            .parse()
            .map_err(|_| format!("invalid variable index in: {label}"))?;

        vals.insert(idx, value);
    }

    if vals.is_empty() {
        return Err("no SOLUTION section found or section is empty".to_string());
    }

    let default = V::default_low::<C>();
    let mut result = vec![default; n_vars];
    for (idx, val) in vals {
        if idx < n_vars {
            result[idx] = val;
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> &'static str {
        "\
HUBO 1
VAR_TYPE BIN
N 10
M 3
OFFSET -1.5
INDEX_BASE 0
META source=test_instance

# term list
0          2.0
0 1       -3.5
2 5 9      0.25
"
    }

    fn unwrap_bin<C: Coeff>(e: HuboInstanceEnum<C>) -> HuboInstance<C, Bin> {
        match e {
            HuboInstanceEnum::Bin(i) => i,
            _ => panic!("expected BIN"),
        }
    }

    fn unwrap_spin<C: Coeff>(e: HuboInstanceEnum<C>) -> HuboInstance<C, Spin> {
        match e {
            HuboInstanceEnum::Spin(i) => i,
            _ => panic!("expected SPIN"),
        }
    }

    #[test]
    fn parse_sample() {
        let (inst, _) = parse::<f64>(sample_input()).unwrap();
        let inst = unwrap_bin(inst);
        assert_eq!(inst.n_vars(), 10);
        assert_eq!(inst.n_terms(), 3);
        assert!((inst.offset - (-1.5)).abs() < f64::EPSILON);
        assert_eq!(inst.terms.len(), 3);
        assert_eq!(inst.terms[0].indices, vec![0]);
        assert!((inst.terms[0].coeff - 2.0).abs() < f64::EPSILON);
        assert_eq!(inst.terms[1].indices, vec![0, 1]);
        assert!((inst.terms[1].coeff - (-3.5)).abs() < f64::EPSILON);
        assert_eq!(inst.terms[2].indices, vec![2, 5, 9]);
        assert!((inst.terms[2].coeff - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_spin_index_base_1() {
        let input = "\
HUBO 1
VAR_TYPE SPIN
N 4
M 2
INDEX_BASE 1

1 2    1.0
3 4   -0.5
";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.terms[0].indices, vec![0, 1]);
        assert_eq!(inst.terms[1].indices, vec![2, 3]);
    }

    #[test]
    fn parse_huso_magic() {
        let input = "\
HUSO 1
VAR_TYPE SPIN
N 3
M 2
0 1.0
1 2 -2.5
";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.n_vars(), 3);
        assert_eq!(inst.n_terms(), 2);
    }

    #[test]
    fn missing_magic() {
        let input = "VAR_TYPE BIN\nN 3\nM 0\n";
        assert!(matches!(
            parse::<f64>(input),
            Err(ParseError::InvalidMagic(_))
        ));
    }

    #[test]
    fn missing_header() {
        let input = "HUBO 1\nVAR_TYPE BIN\nM 0\n";
        assert!(matches!(
            parse::<f64>(input),
            Err(ParseError::MissingHeader("N"))
        ));
    }

    #[test]
    fn duplicate_header() {
        let input = "HUBO 1\nVAR_TYPE BIN\nVAR_TYPE SPIN\nN 3\nM 0\n";
        assert!(matches!(
            parse::<f64>(input),
            Err(ParseError::DuplicateHeader { .. })
        ));
    }

    #[test]
    fn term_count_mismatch() {
        let input = "HUBO 1\nVAR_TYPE BIN\nN 3\nM 2\n0 1.0\n";
        assert!(matches!(
            parse::<f64>(input),
            Err(ParseError::TermCountMismatch {
                declared: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn index_out_of_range() {
        let input = "HUBO 1\nVAR_TYPE BIN\nN 3\nM 1\n5 1.0\n";
        assert!(matches!(
            parse::<f64>(input),
            Err(ParseError::IndexOutOfRange { index: 5, .. })
        ));
    }

    #[test]
    fn duplicate_index_in_term_warns() {
        let input = "HUBO 1\nVAR_TYPE BIN\nN 3\nM 1\n1 1 2.0\n";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_bin(inst);
        assert_eq!(inst.terms[0].indices, vec![1]);
        assert!((inst.terms[0].coeff - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn spin_duplicate_even_removes_index() {
        let input = "HUBO 1\nVAR_TYPE SPIN\nN 3\nM 1\n1 1 2.0\n";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.terms.len(), 0);
        assert_eq!(inst.n_terms(), 0);
        assert!((inst.offset - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn spin_duplicate_odd_keeps_one() {
        let input = "HUBO 1\nVAR_TYPE SPIN\nN 3\nM 1\n1 1 1 2 5.0\n";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.terms.len(), 1);
        assert_eq!(inst.terms[0].indices, vec![1, 2]);
        assert!((inst.terms[0].coeff - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn spin_duplicate_mixed_pairwise() {
        let input = "HUBO 1\nVAR_TYPE SPIN\nN 3\nM 1\n0 0 1 1 2 3.0\n";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.terms.len(), 1);
        assert_eq!(inst.terms[0].indices, vec![2]);
        assert!((inst.terms[0].coeff - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_file() {
        assert!(matches!(parse::<f64>(""), Err(ParseError::EmptyFile)));
        assert!(matches!(
            parse::<f64>("  \n# comment\n"),
            Err(ParseError::EmptyFile)
        ));
    }

    #[test]
    fn offset_defaults_to_zero() {
        let input = "HUBO 1\nVAR_TYPE BIN\nN 3\nM 0\n";
        let (inst, _) = parse::<f64>(input).unwrap();
        let inst = unwrap_bin(inst);
        assert!((inst.offset - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_json_tuple_object() {
        let input = r#"{
            "(1, 2, 3)": 3,
            "(1,)": 7,
            "(2, 3)": 8
        }"#;
        let (inst, _) = parse_json::<f64>(input).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.offset, 0.0);
        assert_eq!(inst.n_vars(), 4);
        assert_eq!(inst.n_terms(), 3);
    }

    #[test]
    fn parse_auto_detects_json() {
        let input = r#"{"(0, 2)": -1.5, "(1,)": 2}"#;
        let (inst, _) = parse_auto::<f64>(input, Some("instance.json")).unwrap();
        let inst = unwrap_spin(inst);
        assert_eq!(inst.n_vars(), 3);
        assert_eq!(inst.n_terms(), 2);
    }
}
