/// Static-rank shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape<const R: usize> {
    pub dims: [usize; R],
}

impl<const R: usize> Shape<R> {
    pub fn new(dims: [usize; R]) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        R
    }

    /// Convert to dynamic shape
    pub fn to_dyn(&self) -> super::shape::ShapeDyn {
        super::shape::ShapeDyn {
            dims: self.dims.to_vec(),
        }
    }
}

/// Dynamic-rank shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeDyn {
    pub dims: Vec<usize>,
}

impl ShapeDyn {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_static_shape() {
        let s = Shape::<2>::new([10, 20]);
        assert_eq!(s.rank(), 2);
        let dyn_s = s.to_dyn();
        assert_eq!(dyn_s.dims, vec![10, 20]);
    }

    #[test]
    fn test_dynamic_shape() {
        let s = ShapeDyn::new(vec![7, 8, 9]);
        assert_eq!(s.rank(), 3);
    }

}

