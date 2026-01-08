
use crate::tensor::{TensorView, TensorViewMut};
use crate::layout::Layout;

/* ============================================================
   BLAS ABI
   ============================================================ */

#[derive(Copy, Clone, Debug)]
pub enum BlasTranspose {
    NoTrans,
    Trans,
}

pub trait BlasBackend {
    fn gemm_f32(
        &self,
        trans_a: BlasTranspose,
        trans_b: BlasTranspose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/* ============================================================
   Layout → BLAS lowering
   ============================================================ */

fn lower_matrix(layout: &Layout) -> (i32, BlasTranspose) {
    assert_eq!(layout.shape().ndim(), 2);

    let s0 = layout.stride().at(0);
    let s1 = layout.stride().at(1);

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

    assert_eq!(la.shape().ndim(), 2);
    assert_eq!(lb.shape().ndim(), 2);
    assert_eq!(lc.shape().ndim(), 2);

    let m = la.shape().at(0) as i32;
    let k = la.shape().at(1) as i32;
    let n = lb.shape().at(1) as i32;

    assert_eq!(lb.shape().at(0) as i32, k);
    assert_eq!(lc.shape().at(0) as i32, m);
    assert_eq!(lc.shape().at(1) as i32, n);

    /* ---------- layout checks ---------- */

    assert!(la.is_contiguous());
    assert!(lb.is_contiguous());
    assert!(lc.is_contiguous());

    /* ---------- BLAS lowering ---------- */

    let (lda, ta) = lower_matrix(la);
    let (ldb, tb) = lower_matrix(lb);
    let ldc = lc.stride().at(0) as i32;

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
    use crate::{Tensor, Layout, Shape, Tuple};

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
        let shape = Shape::new(Tuple::from_vec(vec![2, 2]));

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

