use core::fmt;

/// Zero-sized type representing a compile-time constant dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Const<const N: usize>;

/// Dimension that can be compile-time static or runtime dynamic
///
/// Convention:
/// - `N != 0` → static dimension known at compile time
/// - `N == 0` → dynamic dimension known only at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim<const N: usize> {
    /// Compile-time constant dimension
    Static(Const<N>),

    /// Runtime dimension (only allowed when N == 0)
    Dynamic(usize),
}

impl<const N: usize> Dim<N> {
    /// Construct a compile-time static dimension
    pub const fn static_dim() -> Self {
        // Enforced by convention: Static only meaningful when N != 0
        Dim::Static(Const)
    }

    /// Construct a runtime dynamic dimension
    ///
    /// # Panics
    /// Panics if used when `N != 0`
    pub fn dynamic(v: usize) -> Self {
        assert!(
            N == 0,
            "Dynamic dimension only allowed when const N == 0"
        );
        Dim::Dynamic(v)
    }

    /// Construct a runtime dynamic dimension
    ///
    /// # Panics
    /// Panics if used when `N != 0`
    pub fn to_dynamic(&self) -> Dim<0> {
        assert!(
            N != 0,
            "Only Static dimension can be converted to Dynamic"
        );
        Dim::Dynamic(N)
    }

    /// Return the concrete value of the dimension
    #[inline(always)]
    pub fn value(&self) -> usize {
        match *self {
            Dim::Static(_) => N,
            Dim::Dynamic(v) => v,
        }
    }

    /// Returns true if the dimension is compile-time static
    #[inline(always)]
    pub const fn is_static(&self) -> bool {
        matches!(self, Dim::Static(_))
    }
}

/// Display format:
/// - Static dimensions are prefixed with `_`
/// - Dynamic dimensions print the value
impl<const N: usize> fmt::Display for Dim<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Dim::Static(_) => write!(f, "_{}", N),
            Dim::Dynamic(v) => write!(f, "{}", v),
        }
    }
}

/// Allow easy creation from usize for dynamic dimensions
impl From<usize> for Dim<0> {
    fn from(v: usize) -> Self {
        Dim::dynamic(v)
    }
}

use std::env;

#[cfg(test)]
mod tests {
    use super::*;

    fn read_n(default: usize) -> usize {
        env::var("DIMS_TEST_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
    }
    #[test]
    fn static_dimension_works() {
        let d = Dim::<32>::static_dim();
        assert!(d.is_static());
        assert_eq!(d.value(), 32);
        assert_eq!(format!("{}", d), "_32");
    }

    #[test]
    fn dynamic_dimension_works() {
        let n = read_n(17);
        let d = Dim::<0>::dynamic(n);
        assert!(!d.is_static());
        assert_eq!(d.value(), n);
        assert_eq!(format!("{}", d), "17");
    }

    #[test]
    fn from_usize_for_dynamic() {
        let d: Dim<0> = 42usize.into();
        assert_eq!(d.value(), 42);
    }

    #[test]
    #[should_panic]
    fn dynamic_not_allowed_for_static_const() {
        let _ = Dim::<8>::dynamic(8);
    }

    #[test]
    fn static_is_zero_cost() {
        let d = Dim::<64>::static_dim();
        let v = d.value();
        // This should constant-fold
        assert_eq!(v, 64);
        let k = d.to_dynamic();
        let v = k.value();
        // This may or may NOT constant-fold
        assert_eq!(v, 64);
    }
}

