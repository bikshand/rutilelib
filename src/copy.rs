use crate::tensor::{Tensor, TensorView, TensorViewMut};
use crate::tuple::Tuple;
use crate::shape::Shape;
use crate::layout::*;
use std::ptr::NonNull;

/// Copy from `src` (Tensor / TensorView) to `dst` (Tensor / TensorViewMut)
pub fn tensor_copy<T: Copy>(
    src: &TensorView<'_, T>,
    dst: &mut TensorViewMut<'_, T>,
) {
    let shape = src.layout().shape();
    assert_eq!(shape, dst.layout().shape(), "tensor_copy: shape mismatch");

    // fast path for contiguous 2D tensors
    if src.layout().is_contiguous() && dst.layout().is_contiguous() {
        let n = src.layout().size();
        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr.as_ptr(), dst.ptr.as_ptr(), n);
        }
        return;
    }

    // fallback: strided / N-D copy
    copy_recursive(src, dst, 0, &mut Vec::new());
}

/// Recursive N-D copy
fn copy_recursive<T: Copy>(
    src: &TensorView<'_, T>,
    dst: &mut TensorViewMut<'_, T>,
    dim: usize,
    idx: &mut Vec<usize>,
) {
    let ndims = src.layout().shape().dims.len();
    if dim == ndims {
        let coord = Tuple::Int(idx.clone());
        unsafe {
            *dst.get_mut(&coord) = *src.get(&coord);
        }
        return;
    }

    let dim_len = src.layout().shape().dims.flatten()[dim];
    for i in 0..dim_len {
        idx.push(i);
        copy_recursive(src, dst, dim + 1, idx);
        idx.pop();
    }
}

fn assert_tensor_eq<T: PartialEq + std::fmt::Debug>(
    src: &crate::tensor::Tensor<T>,
    dst: &crate::tensor::Tensor<T>,
) {
    assert_eq!(src.layout().shape(), dst.layout().shape());

    let mut coord = Vec::new();
    enumerate_coords(&src.layout().shape().dims, &mut coord, &mut |c| {
        let t = crate::tuple::Tuple::Int(c.to_vec());
        unsafe {
            assert_eq!(
                src.as_view().get(&t),
                dst.as_view().get(&t),
                "Mismatch at coord {:?}",
                c
            );
        }
    });
}

fn enumerate_coords(
    shape: &crate::tuple::Tuple,
    prefix: &mut Vec<usize>,
    f: &mut impl FnMut(&[usize]),
) {
    match shape {
        crate::tuple::Tuple::Int(dims) => {
            enumerate_int(dims, 0, prefix, f);
        }
        crate::tuple::Tuple::Tup(children) => {
            for child in children {
                enumerate_coords(child, prefix, f);
            }
        }
    }
}

fn enumerate_int(
    dims: &[usize],
    d: usize,
    prefix: &mut Vec<usize>,
    f: &mut impl FnMut(&[usize]),
) {
    if d == dims.len() {
        f(prefix);
        return;
    }

    for i in 0..dims[d] {
        prefix.push(i);
        enumerate_int(dims, d + 1, prefix, f);
        prefix.pop();
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_copy_tensor_to_tensor() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![2]),
            ]),
        );

        let mut t1 = Tensor::new(vec![1, 2, 3, 4], Layout::new::<RowMajor>(shape.clone()));
        let mut t2 = Tensor::new(vec![0;4], Layout::new::<RowMajor>(shape.clone()));

        let src = t1.as_view();
        let mut dst = t2.as_view_mut();
        tensor_copy(&src, &mut dst);

        unsafe {
            for i in 0..4 {
                assert_eq!(*dst.ptr.as_ptr().add(i), *src.ptr.as_ptr().add(i));
            }
        }
    }

    #[test]
    fn copy_hierarchical_contiguous() {

        let shape = Shape::new(Tuple::Tup(vec![
            Tuple::Int(vec![2, 3]),
            Tuple::Int(vec![4]),
        ]));

        let stride = Tuple::Tup(vec![
            Tuple::Int(vec![12, 4]), // (3*4, 4)
            Tuple::Int(vec![1]),
        ]);

        let layout = Layout::with_shape_stride(shape.clone(), stride);

        let src_data: Vec<i32> = (0..24).collect();
        let mut dst_data = vec![0; 24];

        let src = Tensor::new(src_data.clone(), layout.clone());
        let mut dst = Tensor::new(dst_data, layout);

        tensor_copy(&src.as_view(), &mut dst.as_view_mut());

        let src = src.as_view();
        let dst = dst.as_view_mut();
        unsafe {
            for i in 0..24 {
                assert_eq!(*dst.ptr.as_ptr().add(i), *src.ptr.as_ptr().add(i));
            }
        }
    }

}

