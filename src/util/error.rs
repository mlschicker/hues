use std::fmt;

/// Errors that can occur while parsing a HUBO-TL file.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// The file is empty or contains only comments/blank lines.
    EmptyFile,
    /// The JSON instance format is invalid.
    InvalidJson(String),
    /// The first non-empty, non-comment line is not `HUBO 1`.
    InvalidMagic(String),
    /// A required header keyword is missing.
    MissingHeader(&'static str),
    /// A header keyword appeared more than once.
    DuplicateHeader { keyword: String, line: usize },
    /// A header keyword has an invalid value.
    InvalidHeaderValue {
        keyword: String,
        value: String,
        line: usize,
    },
    /// An unknown keyword was encountered where a header was expected.
    UnknownKeyword { keyword: String, line: usize },
    /// A variable index is out of the valid range.
    IndexOutOfRange { index: u64, line: usize },
    /// A term line has no variable indices (only a coefficient).
    EmptyTerm { line: usize },
    /// Could not parse a token as a coefficient (integer or float).
    InvalidCoeff { token: String, line: usize },
    /// Could not parse a token as an index.
    InvalidIdx { token: String, line: usize },
    /// The number of term lines does not match the declared `M`.
    TermCountMismatch { declared: usize, actual: usize },
    /// A term line has too few tokens (needs at least an index and a coefficient).
    InsufficientTokens { line: usize },
    /// A META line has an invalid format.
    InvalidMeta { line: usize, detail: String },
    /// A META line appeared in the term body (only allowed in the header).
    MetaInBody { line: usize },
    /// The index range seen in the term body does not match the declared number of variables.
    IndexRangeMismatch { lo: u64, hi: u64, n_vars: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::EmptyFile => write!(f, "file is empty or contains no meaningful lines"),
            ParseError::InvalidJson(detail) => write!(f, "invalid JSON instance: {detail}"),
            ParseError::InvalidMagic(got) => {
                write!(f, "expected first line to be `HUBO 1`, got `{got}`")
            }
            ParseError::MissingHeader(kw) => write!(f, "missing required header `{kw}`"),
            ParseError::DuplicateHeader { keyword, line } => {
                write!(f, "line {line}: duplicate header keyword `{keyword}`")
            }
            ParseError::InvalidHeaderValue {
                keyword,
                value,
                line,
            } => write!(f, "line {line}: invalid value `{value}` for `{keyword}`"),
            ParseError::UnknownKeyword { keyword, line } => {
                write!(f, "line {line}: unknown keyword `{keyword}`")
            }
            ParseError::IndexOutOfRange { index, line } => {
                write!(f, "line {line}: variable index {index} is out of range")
            }
            ParseError::EmptyTerm { line } => {
                write!(f, "line {line}: term line has no variable indices")
            }
            ParseError::InvalidCoeff { token, line } => {
                write!(f, "line {line}: cannot parse `{token}` as coefficient")
            }
            ParseError::InvalidIdx { token, line } => {
                write!(f, "line {line}: cannot parse `{token}` as integer")
            }
            ParseError::TermCountMismatch { declared, actual } => {
                write!(
                    f,
                    "declared M={declared} terms but found {actual} term lines"
                )
            }
            ParseError::InsufficientTokens { line } => {
                write!(f, "line {line}: too few tokens on term line")
            }
            ParseError::InvalidMeta { line, detail } => {
                write!(f, "line {line}: invalid META: {detail}")
            }
            ParseError::MetaInBody { line } => {
                write!(
                    f,
                    "line {line}: META is only allowed in the header, not in the term body"
                )
            }
            ParseError::IndexRangeMismatch { lo, hi, n_vars } => {
                write!(
                    f,
                    "index range mismatch: expected {n_vars} variables but found {lo}..{hi}"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Non-fatal warnings emitted during parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseWarning {
    /// A term line contains a duplicate variable index (the duplicate was dropped).
    DuplicateIndex { index: u64, line: usize },
    /// The input variables are 1 indexed
    OneIndexed { line: usize },
    /// The index base is ambiguous (min index > 0 but max index < N); assumed 0-based.
    AmbiguousIndexBase,
}

impl fmt::Display for ParseWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseWarning::DuplicateIndex { index, line } => {
                write!(
                    f,
                    "line {line}: duplicate variable index {index} in term (duplicate dropped)"
                )
            }
            ParseWarning::OneIndexed { line } => {
                write!(
                    f,
                    "line {line}: index exceeds range by one, assuming input variables are 1 indexed (indexes should start at 0)"
                )
            }
            ParseWarning::AmbiguousIndexBase => {
                write!(
                    f,
                    "index base is ambiguous (lowest index > 0 but highest index < N); assuming 0-based indexing"
                )
            }
        }
    }
}
