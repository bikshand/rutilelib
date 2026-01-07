/// Static-rank tuple
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tuple<const R: usize> {
    pub values: [usize; R],
}

impl<const R: usize> Tuple<R> {
    pub fn new(values: [usize; R]) -> Self {
        Self { values }
    }

    pub fn rank(&self) -> usize {
        R
    }

    /// Convert to dynamic tuple
    pub fn to_dyn(&self) -> super::tuple::TupleDyn {
        super::tuple::TupleDyn {
            values: self.values.to_vec(),
        }
    }
}

/// Dynamic-rank tuple
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupleDyn {
    pub values: Vec<usize>,
}

impl TupleDyn {
    pub fn new(values: Vec<usize>) -> Self {
        Self { values }
    }

    pub fn rank(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_tuple() {
        let t = Tuple::<3>::new([1, 2, 3]);
        assert_eq!(t.rank(), 3);
        let dyn_t = t.to_dyn();
        assert_eq!(dyn_t.values, vec![1, 2, 3]);
    }

    #[test]
    fn test_dynamic_tuple() {
        let t = TupleDyn::new(vec![5, 6, 7, 8]);
        assert_eq!(t.rank(), 4);
    }
}

