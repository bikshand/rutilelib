
use crate::tensor::{Tensor, TensorView, TensorViewMut};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::tuple::Tuple;
use crate::blas::*;

/* ============================================================
   Layout → BLAS lowering
   ============================================================ */

fn lower_matrix(layout: &Layout) -> (i32, BlasTranspose) {
    assert_eq!(layout.shape().flat_len(), 2);

    let s0 = layout.stride().flat_at(0);
    let s1 = layout.stride().flat_at(1);

    // Row-major: [i][j] → j is contiguous
    if s1 == 1 {
        (s0 as i32, BlasTranspose::NoTrans)
    }
    // Column-major: transpose trick
    else if s0 == 1 {
        (s1 as i32, BlasTranspose::Trans)
    }
    else {
        panic!("GEMM requires dense 2D layout");
    }
}

/* ============================================================
   Public GEMM API
   ============================================================ */

pub fn gemm_f32<B: BlasBackend>(
    backend: &B,
    a: &TensorView<'_, f32>,
    b: &TensorView<'_, f32>,
    c: &mut TensorViewMut<'_, f32>,
) {
    let la = a.layout();
    let lb = b.layout();
    let lc = c.layout();

    /* ---------- shape checks ---------- */

    assert_eq!(la.shape().flat_len(), 2);
    assert_eq!(lb.shape().flat_len(), 2);
    assert_eq!(lc.shape().flat_len(), 2);

    let m = la.shape().flat_at(0) as i32;
    let k = la.shape().flat_at(1) as i32;
    let n = lb.shape().flat_at(1) as i32;

    assert_eq!(lb.shape().flat_at(0) as i32, k);
    assert_eq!(lc.shape().flat_at(0) as i32, m);
    assert_eq!(lc.shape().flat_at(1) as i32, n);

    /* ---------- layout checks ---------- */

    assert!(la.is_contiguous());
    assert!(lb.is_contiguous());
    assert!(lc.is_contiguous());

    /* ---------- BLAS lowering ---------- */

    let (lda, ta) = lower_matrix(la);
    let (ldb, tb) = lower_matrix(lb);
    let ldc = lc.stride().flat_at(0) as i32;

    unsafe {
        backend.gemm_f32(
            ta,
            tb,
            m,
            n,
            k,
            1.0,
            a.ptr.as_ptr(),
            lda,
            b.ptr.as_ptr(),
            ldb,
            0.0,
            c.ptr.as_ptr(),
            ldc,
        );
    }
}

/* ============================================================
   Mock backend for unit testing
   ============================================================ */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::layout::Layout;
    use crate::shape::Shape;
    use crate::tuple::Tuple;

    struct MockBlas;

    impl BlasBackend for MockBlas {
        fn gemm_f32(
            &self,
            _ta: BlasTranspose,
            _tb: BlasTranspose,
            m: i32,
            n: i32,
            k: i32,
            _alpha: f32,
            _a: *const f32,
            lda: i32,
            _b: *const f32,
            ldb: i32,
            _beta: f32,
            _c: *mut f32,
            ldc: i32,
        ) {
            // Validate lowering only
            assert_eq!(m, 2);
            assert_eq!(n, 2);
            assert_eq!(k, 2);
            assert!(lda > 0);
            assert!(ldb > 0);
            assert!(ldc > 0);
        }
    }

    #[test]
    fn gemm_dispatch_only() {
        let shape = Shape::new(Tuple::int(vec![2, 2]));

        let a = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            Layout::row_major(shape.clone()),
        );

        let b = Tensor::new(
            vec![5.0, 6.0, 7.0, 8.0],
            Layout::row_major(shape.clone()),
        );

        let mut c = Tensor::new(
            vec![0.0; 4],
            Layout::row_major(shape),
        );

        let backend = MockBlas;

        gemm_f32(&backend, &a.as_view(), &b.as_view(), &mut c.as_view_mut());
    }
}


#[test]
fn gemm_2x2_row_major_correctness() {
    let shape = Shape::new(Tuple::int(vec![2, 2]));

    let a = Tensor::new(
        vec![1.0, 2.0,
             3.0, 4.0],
        Layout::row_major(shape.clone()),
    );

    let b = Tensor::new(
        vec![5.0, 6.0,
             7.0, 8.0],
        Layout::row_major(shape.clone()),
    );

    let mut c = Tensor::new(
        vec![0.0; 4],
        Layout::row_major(shape),
    );

    let backend = GenericBlas;

    gemm_f32(&backend, &a.as_view(), &b.as_view(), &mut c.as_view_mut());

    // Expected:
    // [ 1*5 + 2*7 , 1*6 + 2*8 ]
    // [ 3*5 + 4*7 , 3*6 + 4*8 ]
    let expected = vec![19.0, 22.0, 43.0, 50.0];

    unsafe {
        for i in 0..4 {
            assert_eq!(*c.as_view().ptr.as_ptr().add(i), expected[i]);
        }
    }
}

