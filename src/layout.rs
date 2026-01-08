use crate::shape::Shape;
use crate::tuple::Tuple;

/// Layout = mapping from coordinates → linear index
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    stride: Tuple,
}

/// Layout policy trait
pub trait LayoutPolicy {
    fn make_stride(shape: &Shape) -> Tuple;
}

/// Default row-major policy
pub struct RowMajor;

/// Column-major policy
pub struct ColMajor;

impl Layout {
    pub fn new<P: LayoutPolicy>(shape: Shape) -> Self {
        let stride = P::make_stride(&shape);
        Self { shape, stride }
    }

    pub fn row_major(shape: Shape) -> Self {
        Layout::new::<RowMajor>(shape)
    }

    pub fn col_major(shape: Shape) -> Self {
        Layout::new::<ColMajor>(shape)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &Tuple {
        &self.stride
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Maximum linear index + 1 = codomain size
    pub fn cosize(&self) -> usize {
        fn recur(shape: &Tuple, stride: &Tuple) -> usize {
            match (shape, stride) {
                (Tuple::Int(sizes), Tuple::Int(strides)) => {
                    let last_idx = sizes.iter().zip(strides.iter())
                                        .map(|(s, st)| (s - 1) * st)
                                        .sum::<usize>();
                    last_idx + 1
                }
                (Tuple::Tup(ss), Tuple::Tup(sts)) => {
                    let last_s = ss.last().expect("Empty Tup in cosize");
                    let last_st = sts.last().expect("Stride mismatch in cosize");
                    recur(last_s, last_st)
                }
                _ => panic!("Shape/Stride mismatch in cosize"),
            }
        }

        recur(&self.shape.dims, &self.stride)
    }

    pub fn crd2idx(&self, crd: &Tuple) -> usize {
        crd.dot(&self.stride)
    }

    pub fn idx2crd(&self, mut idx: usize) -> Tuple {
        fn recur(idx: &mut usize, shape: &Tuple, stride: &Tuple) -> Tuple {
            match (shape, stride) {
                (Tuple::Int(sizes), Tuple::Int(strides)) => {
                    let mut out = Vec::with_capacity(sizes.len());
                    for (sz, st) in sizes.iter().zip(strides.iter()) {
                        let v = *idx / *st;
                        *idx -= v * st;
                        assert!(v < *sz);
                        out.push(v);
                    }
                    Tuple::Int(out)
                }
                (Tuple::Tup(ss), Tuple::Tup(st)) => {
                    Tuple::Tup(
                        ss.iter()
                          .zip(st.iter())
                          .map(|(s, t)| recur(idx, s, t))
                          .collect()
                    )
                }
                _ => panic!("Layout mismatch in idx2crd"),
            }
        }

        recur(&mut idx, &self.shape.dims, &self.stride)
    }
}

impl Layout {
    /// Create a new layout from shape + stride (used for subviews)
    pub(crate) fn with_shape_stride(shape: Shape, stride: Tuple) -> Self {
        Layout { shape, stride }
    }
}

/* ---------- stride helpers ---------- */

fn scale_stride(t: &Tuple, factor: usize) -> Tuple {
    match t {
        Tuple::Int(v) => Tuple::Int(v.iter().map(|x| x * factor).collect()),
        Tuple::Tup(v) => Tuple::Tup(v.iter().map(|x| scale_stride(x, factor)).collect()),
    }
}

fn row_major_stride(shape: &Tuple) -> Tuple {
    match shape {
        Tuple::Int(sizes) => {
            let mut stride = vec![0; sizes.len()];
            let mut acc = 1;
            for i in (0..sizes.len()).rev() {
                stride[i] = acc;
                acc *= sizes[i];
            }
            Tuple::Int(stride)
        }
        Tuple::Tup(v) => {
            let mut acc = 1;
            let mut out = Vec::with_capacity(v.len());
            for s in v.iter().rev() {
                let st = scale_stride(&row_major_stride(s), acc);
                acc *= s.size();
                out.push(st);
            }
            out.reverse();
            Tuple::Tup(out)
        }
    }
}

fn col_major_stride(shape: &Tuple) -> Tuple {
    match shape {
        Tuple::Int(sizes) => {
            let mut stride = Vec::with_capacity(sizes.len());
            let mut acc = 1;
            for s in sizes {
                stride.push(acc);
                acc *= *s;
            }
            Tuple::Int(stride)
        }
        Tuple::Tup(v) => {
            let mut acc = 1;
            let mut out = Vec::with_capacity(v.len());
            for s in v {
                let st = scale_stride(&col_major_stride(s), acc);
                acc *= s.size();
                out.push(st);
            }
            Tuple::Tup(out)
        }
    }
}

/* ---------- policies ---------- */

impl LayoutPolicy for RowMajor {
    fn make_stride(shape: &Shape) -> Tuple {
        row_major_stride(&shape.dims)
    }
}

impl LayoutPolicy for ColMajor {
    fn make_stride(shape: &Shape) -> Tuple {
        col_major_stride(&shape.dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::Tuple;

    #[test]
    fn row_major_roundtrip() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);

        for i in 0..6 {
            let crd = layout.idx2crd(i);
            let idx = layout.crd2idx(&crd);
            assert_eq!(i, idx);
        }
    }

    #[test]
    fn hierarchical_stride_correct() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::tup(vec![
                Tuple::int(vec![3]),
                Tuple::int(vec![4]),
            ]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);
        assert_eq!(layout.stride().to_string(), "(12,(4,1))");
    }
}


