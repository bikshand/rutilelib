use std::fmt;

/// Recursive integer tuple (CuTe-style)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tuple {
    Int(usize),
    Tup(Vec<Tuple>),
}

impl Tuple {
    pub fn int(v: usize) -> Self {
        Tuple::Int(v)
    }

    pub fn tup(v: Vec<Tuple>) -> Self {
        Tuple::Tup(v)
    }

    /// Total number of leaf elements
    pub fn size(&self) -> usize {
        match self {
            Tuple::Int(v) => *v,
            Tuple::Tup(vs) => vs.iter().map(|t| t.size()).product(),
        }
    }

    /// Depth of the tuple
    pub fn depth(&self) -> usize {
        match self {
            Tuple::Int(_) => 0,
            Tuple::Tup(vs) => 1 + vs.iter().map(|t| t.depth()).max().unwrap_or(0),
        }
    }
}

impl fmt::Display for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tuple::Int(v) => write!(f, "{v}"),
            Tuple::Tup(vs) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_leaf() {
        let t = Tuple::int(5);
        assert_eq!(t.size(), 5);
        assert_eq!(t.depth(), 0);
        assert_eq!(t.to_string(), "5");
    }

    #[test]
    fn tuple_nested() {
        let t = Tuple::tup(vec![
            Tuple::int(2),
            Tuple::tup(vec![Tuple::int(3), Tuple::int(4)]),
        ]);

        assert_eq!(t.size(), 2 * 3 * 4);
        assert_eq!(t.depth(), 2);
        assert_eq!(t.to_string(), "(2,(3,4))");
    }
}

