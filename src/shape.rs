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
    
    pub fn flat_at(&self, i: usize) -> usize {
          self.dims.flat_at(i)
    }

    pub fn flat_len(&self) -> usize {
          self.dims.flat_len()
    }


    /// Depth of hierarchy
    pub fn depth(&self) -> usize {
        self.dims.depth()
    }

    /// Hierarchical get
    pub fn get(&self, index: usize) -> Shape {
        Shape {
            dims: self.dims.get(index),
        }
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
            Tuple::int1(4),
            Tuple::int1(5),
        ]));

        assert_eq!(s.size(), 20);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.depth(), 1);
        assert_eq!(s.to_string(), "(4,5)");
    }

    #[test]
    fn shape_hierarchical() {
        let s = Shape::new(Tuple::tup(vec![
            Tuple::int1(2),
            Tuple::tup(vec![Tuple::int1(3), Tuple::int1(4)]),
        ]));

        assert_eq!(s.size(), 24);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.depth(), 2);
        assert_eq!(s.to_string(), "(2,(3,4))");
    }

    #[test]
    fn shape_get() {
        let s = Shape::new(Tuple::tup(vec![
            Tuple::int1(2),
            Tuple::int1(3),
            Tuple::tup(vec![Tuple::int1(4), Tuple::int1(2)]),
        ]));

        let s1 = s.get(1);
        assert_eq!(s1.size(), 3);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s1.to_string(), "3");

        let s1 = s.get(2);
        assert_eq!(s1.size(), 8);
        assert_eq!(s1.rank(), 2);
        assert_eq!(s1.to_string(), "(4,2)");
    }
}

