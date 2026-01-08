use std::fmt;

/// Recursive integer tuple (CuTe-style)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tuple {
    Int(Vec<usize>),
    Tup(Vec<Tuple>),
}

impl Tuple {
    /// Create Int tuple from a Vec
    pub fn int(v: Vec<usize>) -> Self {
        Tuple::Int(v)
    }

    /// Create Int tuple from a single usize
    pub fn int1(x: usize) -> Self {
        Tuple::Int(vec![x])
    }

    /// Create Tup tuple from Vec<Tuple>
    pub fn tup(v: Vec<Tuple>) -> Self {
        Tuple::Tup(v)
    }

    /// Total number of leaf elements
    pub fn size(&self) -> usize {
        match self {
            Tuple::Int(v) => v.iter().product(),
            Tuple::Tup(vs) => vs.iter().map(|t| t.size()).product(),
        }
    }

    /// Number of flattened elements
    pub fn flat_len(&self) -> usize {
        self.flatten().len()
    }

    /// Access the i-th element in flattened form
    pub fn flat_at(&self, i: usize) -> usize {
        self.flatten()[i]
    }

    /// Depth of the tuple
    pub fn depth(&self) -> usize {
        match self {
            Tuple::Int(_) => 0,
            Tuple::Tup(vs) => 1 + vs.iter().map(|t| t.depth()).max().unwrap_or(0),
        }
    }

    /// Length of the tuple (number of elements at top level)
    pub fn len(&self) -> usize {
        match self {
            Tuple::Int(v) => v.len(),
            Tuple::Tup(v) => v.len(),
        }
    }

    /// Hierarchical get: returns ith element as Tuple
    pub fn get(&self, index: usize) -> Tuple {
        match self {
            Tuple::Int(v) => {
                assert!(index < v.len(), "Index out of bounds in Tuple::get");
                Tuple::Int(vec![v[index]])
            }
            Tuple::Tup(v) => {
                assert!(index < v.len(), "Index out of bounds in Tuple::get");
                v[index].clone()
            }
        }
    }

    /// Recursive flattened iterator over leaf integers
    pub fn iter_flat(&self) -> Box<dyn Iterator<Item = &usize> + '_> {
        match self {
            Tuple::Int(v) => Box::new(v.iter()),
            Tuple::Tup(vs) => Box::new(vs.iter().flat_map(|t| t.iter_flat())),
        }
    }

    /// Hierarchical dot product
    pub fn dot(&self, other: &Tuple) -> usize {
        match (self, other) {
            (Tuple::Int(a), Tuple::Int(b)) => {
                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            }
            (Tuple::Tup(a), Tuple::Tup(b)) => {
                a.iter().zip(b.iter()).map(|(x, y)| x.dot(y)).sum()
            }
            _ => panic!("Tuple mismatch in dot"),
        }
    }
}

impl Tuple {
    /// Concatenate two tuples at top level
    pub fn concat(lhs: &Tuple, rhs: &Tuple) -> Tuple {
        match (lhs, rhs) {
            (Tuple::Tup(a), Tuple::Tup(b)) => {
                let mut v = a.clone();
                v.extend(b.clone());
                Tuple::Tup(v)
            }
            _ => panic!("Tuple::concat requires Tup + Tup"),
        }
    }

    /// Flatten tuple into leaf extents (row-major order)
    pub fn flatten(&self) -> Vec<usize> {
        match self {
            Tuple::Int(v) => v.clone(),
            Tuple::Tup(vs) => vs.iter().flat_map(|t| t.flatten()).collect(),
        }
    }

    /// Product of all leaves
    pub fn product(&self) -> usize {
        self.flatten().into_iter().product()
    }

    /// Element-wise divide (strict)
    pub fn div_exact(&self, rhs: &Tuple) -> Tuple {
        let a = self.flatten();
        let b = rhs.flatten();
        assert_eq!(a.len(), b.len(), "Tuple::div_exact rank mismatch");

        Tuple::Int(
            a.into_iter()
                .zip(b.into_iter())
                .map(|(x, y)| {
                    assert!(x % y == 0, "Tuple::div_exact not divisible");
                    x / y
                })
                .collect(),
        )
    }

    /// Element-wise modulo
    pub fn mod_exact(&self, rhs: &Tuple) -> Tuple {
        let a = self.flatten();
        let b = rhs.flatten();
        assert_eq!(a.len(), b.len(), "Tuple::mod_exact rank mismatch");

        Tuple::Int(a.into_iter().zip(b).map(|(x, y)| x % y).collect())
    }
}

impl fmt::Display for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tuple::Int(v) => {
                if v.len() == 1 {
                    write!(f, "{}", v[0])
                } else {
                    write!(f, "(")?;
                    for (i, v) in v.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, ")")
                }
            }
            Tuple::Tup(vs) => {
                if vs.len() == 1 {
                    write!(f, "{}", vs[0])
                } else {
                    write!(f, "(")?;
                    for (i, v) in vs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}

pub type Stride = Tuple;
pub type Indices = Tuple;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_single_int() {
        let t = Tuple::int1(5);
        assert_eq!(t.size(), 5);
        assert_eq!(t.depth(), 0);
        assert_eq!(t.to_string(), "5");
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(0), Tuple::int1(5));
    }

    #[test]
    fn tuple_vec_int() {
        let t = Tuple::int(vec![2, 3, 4]);
        assert_eq!(t.size(), 24); // 2*3*4
        assert_eq!(t.depth(), 0);
        assert_eq!(t.to_string(), "(2,3,4)");
        assert_eq!(t.len(), 3);
        assert_eq!(t.get(1), Tuple::int1(3));
    }

    #[test]
    fn tuple_nested() {
        let t = Tuple::tup(vec![
            Tuple::int1(2),
            Tuple::tup(vec![Tuple::int1(3), Tuple::int1(4)]),
        ]);

        assert_eq!(t.size(), 2 * 3 * 4);
        assert_eq!(t.depth(), 2);
        assert_eq!(t.to_string(), "(2,(3,4))");
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(1).to_string(), "(3,4)");
    }

    #[test]
    fn tuple_iter_flat() {
        let t = Tuple::tup(vec![
            Tuple::int(vec![2, 3]),
            Tuple::int(vec![4, 5]),
        ]);
        let elems: Vec<usize> = t.iter_flat().copied().collect();
        assert_eq!(elems, vec![2,3,4,5]);
    }
}

