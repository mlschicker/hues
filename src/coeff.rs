//! Generic coefficient type for HUBO instances.
//!
//! The [`Coeff`] trait abstracts over the numeric type used for term
//! coefficients and objective values.  Implementing it for both [`f64`]
//! (floating-point) and [`i64`] (exact integer) allows the same solver
//! code to work in either domain, eliminating floating-point rounding
//! errors for pure-integer problems.

use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use num::traits::MulAddAssign;
use num::{Bounded, Num, Signed};

// ---------------------------------------------------------------------------
// Trait definition
// ---------------------------------------------------------------------------

/// A numeric type that can serve as a HUBO coefficient.
///
/// Implemented for [`f64`] and [`i64`].
pub trait Coeff:
    Num
    + Debug
    + Display
    + Signed
    + Bounded
    + PartialOrd
    + PartialEq
    + Clone
    + MulAddAssign
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Product
    + Sum
    + Copy
    + Send
    + Sync
    + 'static
{
    /// Convert to `f64` (potentially lossy for very large `i64` values).
    fn to_f64(self) -> f64;

    /// Create a value from an `i64` literal.
    fn from_i64(v: i64) -> Self;

    /// Parse a coefficient from a string token (as it appears in a HUBO-TL
    /// file).
    fn parse_str(s: &str) -> Result<Self, String>;

    /// Short human-readable name for this type (`"i64"` or `"f64"`).
    fn type_name() -> &'static str;

    /// Convert an `f64` lower-bound value to `Self` in a **conservative** way:
    /// the returned value is ≤ the true lower bound in the `Self` domain.
    ///
    /// * For `f64`: returns the value unchanged.
    /// * For `i64`: returns `⌊f⌋` cast to `i64` (safe: the true integer
    ///   optimal is always ≥ our f64 lower bound, so flooring is conservative).
    fn from_f64_lb(f: f64) -> Self;

    /// Exact integer representation, when this coefficient type has one.
    fn to_i128_exact(self) -> Option<i128>;

    /// Checked conversion from an exact integer representation.
    fn from_i128_checked(v: i128) -> Option<Self>;

    /// Round a lower bound up to the next value on the objective-value grid.
    fn ceil_to_grid(lb: Self, base: Self, granularity: Self) -> Self;

    #[inline]
    fn max_of(self, other: Self) -> Self {
        if self > other { self } else { other }
    }
}

// ---------------------------------------------------------------------------
// f64 implementation
// ---------------------------------------------------------------------------

impl Coeff for f64 {
    // const ZERO: Self = 0.0;
    // const ONE: Self = 1.0;
    // const NEG_ONE: Self = -1.0;
    // const MAX: Self = f64::INFINITY;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        v as f64
    }

    fn parse_str(s: &str) -> Result<Self, String> {
        s.parse::<f64>()
            .map_err(|_| format!("cannot parse `{s}` as float"))
    }

    fn type_name() -> &'static str {
        "f64"
    }

    #[inline]
    fn from_f64_lb(f: f64) -> Self {
        f
    }

    #[inline]
    fn to_i128_exact(self) -> Option<i128> {
        None
    }

    #[inline]
    fn from_i128_checked(_v: i128) -> Option<Self> {
        None
    }

    #[inline]
    fn ceil_to_grid(lb: Self, _base: Self, _granularity: Self) -> Self {
        lb
    }
}

// ---------------------------------------------------------------------------
// i64 implementation
// ---------------------------------------------------------------------------

impl Coeff for i64 {
    // const ZERO: Self = 0;
    // const ONE: Self = 1;
    // const NEG_ONE: Self = -1;
    // const MAX: Self = i64::MAX;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        v
    }

    /// Parse an integer coefficient.
    ///
    /// Accepts plain integer literals (`"42"`, `"-3"`) as well as
    /// float-formatted values that happen to be whole numbers
    /// (`"2.0"`, `"-1e2"`).
    fn parse_str(s: &str) -> Result<Self, String> {
        // Fast path: plain integer
        if let Ok(v) = s.parse::<i64>() {
            return Ok(v);
        }
        // Slow path: float notation that represents an exact integer
        if let Ok(v) = s.parse::<f64>()
            && v.is_finite()
            && v == v.round()
            && v >= i64::MIN as f64
            && v <= i64::MAX as f64
        {
            return Ok(v as i64);
        }
        Err(format!("cannot parse `{s}` as integer"))
    }

    fn type_name() -> &'static str {
        "i64"
    }

    #[inline]
    fn from_f64_lb(f: f64) -> Self {
        // Floor gives a value ≤ f, which is ≤ true lb.
        // Clamp to i64 range to avoid overflow.
        if !f.is_finite() {
            return if f < 0.0 { i64::MIN } else { i64::MAX };
        }
        f.floor() as i64
    }

    #[inline]
    fn to_i128_exact(self) -> Option<i128> {
        Some(self as i128)
    }

    #[inline]
    fn from_i128_checked(v: i128) -> Option<Self> {
        i64::try_from(v).ok()
    }

    #[inline]
    fn ceil_to_grid(lb: Self, base: Self, granularity: Self) -> Self {
        if granularity <= 1 {
            return lb;
        }
        let g = granularity as i128;
        let lb = lb as i128;
        let base = base as i128;
        let delta = (lb - base).rem_euclid(g);
        let rounded = if delta == 0 { lb } else { lb + (g - delta) };
        i64::try_from(rounded).unwrap_or(i64::MAX)
        // return lb as Self;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_parse() {
        assert_eq!(f64::parse_str("2.5"), Ok(2.5));
        assert_eq!(f64::parse_str("-3"), Ok(-3.0));
        assert_eq!(f64::parse_str("1e-3"), Ok(0.001));
        assert!(f64::parse_str("abc").is_err());
    }

    #[test]
    fn i64_parse_plain() {
        assert_eq!(i64::parse_str("42"), Ok(42));
        assert_eq!(i64::parse_str("-3"), Ok(-3));
        assert_eq!(i64::parse_str("0"), Ok(0));
    }

    #[test]
    fn i64_parse_float_notation() {
        assert_eq!(i64::parse_str("2.0"), Ok(2));
        assert_eq!(i64::parse_str("-1.0"), Ok(-1));
        assert_eq!(i64::parse_str("1e2"), Ok(100));
        assert_eq!(i64::parse_str("-1e2"), Ok(-100));
    }

    #[test]
    fn i64_rejects_non_integer() {
        assert!(i64::parse_str("2.5").is_err());
        assert!(i64::parse_str("0.1").is_err());
        assert!(i64::parse_str("abc").is_err());
    }

    // #[test]
    // fn constants() {
    //     assert_eq!(f64::, 0.0);
    //     assert_eq!(f64::one(), 1.0);
    //     assert_eq!(f64::neg_one(), -1.0);

    //     assert_eq!(i64::zero(), 0);
    //     assert_eq!(i64::one(), 1);
    //     assert_eq!(i64::neg_one(), -1);
    // }

    #[test]
    fn arithmetic() {
        fn check<C: Coeff>() {
            let a = C::from_i64(3);
            let b = C::from_i64(2);
            assert_eq!(a + b, C::from_i64(5));
            assert_eq!(a - b, C::from_i64(1));
            assert_eq!(a * b, C::from_i64(6));
            assert_eq!(-a, C::from_i64(-3));
        }
        check::<f64>();
        check::<i64>();
    }
}
