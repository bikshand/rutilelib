/// Static-rank indices
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indices<const R: usize> {
    pub values: [usize; R],
}

impl<const R: usize> Indices<R> {
    pub fn new(values: [usize; R]) -> Self {
        Self { values }
    }

    pub fn zero() -> Self {
        Self { values: [0; R] }
    }

    pub fn rank(&self) -> usize {
        R
    }

    /// Convert to dynamic indices
    pub fn to_dyn(&self) -> super::indices::IndicesDyn {
        super::indices::IndicesDyn {
            values: self.values.to_vec(),
        }
    }
}

/// Dynamic-rank indices
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndicesDyn {
    pub values: Vec<usize>,
}

impl IndicesDyn {
    pub fn new(values: Vec<usize>) -> Self {
        Self { values }
    }

    pub fn zero(rank: usize) -> Self {
        Self { values: vec![0; rank] }
    }

    pub fn rank(&self) -> usize {
        self.values.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_indices() {
        let idx = Indices::<3>::new([1, 2, 3]);
        assert_eq!(idx.rank(), 3);
        let dyn_idx = idx.to_dyn();
        assert_eq!(dyn_idx.values, vec![1, 2, 3]);
    }

    #[test]
    fn test_dynamic_indices() {
        let idx = IndicesDyn::new(vec![4, 5]);
        assert_eq!(idx.rank(), 2);
    }
}

