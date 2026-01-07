use crate::tuple::Tuple;

/// Shape wraps Tuple and adds semantic meaning
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Tuple,
}

impl Shape {
    pub fn new(dims: Tuple) -> Self {
        Self { dims }
    }

    /// Total number of elements
    pub fn size(&self) -> usize {
        self.dims.size()
    }

    /// Rank = number of top-level modes
    pub fn rank(&self) -> usize {
        match &self.dims {
            Tuple::Int(_) => 1,
            Tuple::Tup(v) => v.len(),
        }
    }

    /// Depth of hierarchy
    pub fn depth(&self) -> usize {
        self.dims.depth()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::Tuple;

    #[test]
    fn shape_flat() {
        let s = Shape::new(Tuple::tup(vec![
            Tuple::int(4),
            Tuple::int(5),
        ]));

        assert_eq!(s.size(), 20);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.depth(), 1);
        assert_eq!(s.to_string(), "(4,5)");
    }

    #[test]
    fn shape_hierarchical() {
        let s = Shape::new(Tuple::tup(vec![
            Tuple::int(2),
            Tuple::tup(vec![Tuple::int(3), Tuple::int(4)]),
        ]));

        assert_eq!(s.size(), 24);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.depth(), 2);
        assert_eq!(s.to_string(), "(2,(3,4))");
    }
}

