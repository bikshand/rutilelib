// src/stride/stride.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stride<const R: usize> {
    pub strides: [usize; R],
}

impl<const R: usize> Stride<R> {
    /// Create a new static stride
    pub const fn new(strides: [usize; R]) -> Self {
        Self { strides }
    }

    /// Get stride as slice
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    /// Rank of the stride
    pub const fn rank(&self) -> usize {
        R
    }

    /// Product of all stride values (for bounds)
    pub fn size(&self) -> usize {
        let mut prod = 1;
        let mut i = 0;
        while i < R {
            prod *= self.strides[i];
            i += 1;
        }
        prod
    }

    /// Convert to dynamic stride
    pub fn to_dynamic(&self) -> crate::stride::StrideDyn {
        crate::stride::StrideDyn {
            strides: self.strides.to_vec(),
        }
    }
}


// src/stride/stride_dyn.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrideDyn {
    pub strides: Vec<usize>,
}

impl StrideDyn {
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }

    pub fn rank(&self) -> usize {
        self.strides.len()
    }

    pub fn size(&self) -> usize {
        self.strides.iter().product()
    }
}

// src/tests/stride_tests.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_stride() {
        const S: Stride<3> = Stride::new([1, 2, 6]);
        assert_eq!(S.rank(), 3);
        assert_eq!(S.size(), 12);

        let dyn_stride = S.to_dynamic();
        assert_eq!(dyn_stride.strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_dynamic_stride() {
        let d = StrideDyn::new(vec![1, 2, 6]);
        assert_eq!(d.rank(), 3);
        assert_eq!(d.size(), 12);
    }
}

