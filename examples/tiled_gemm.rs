use rutilelib::tensor::{Tensor, TensorView, TensorViewMut};
use rutilelib::layout::Layout;
use rutilelib::shape::Shape;
use rutilelib::tuple::Tuple;
use rutilelib::gemm::{gemm_f32, assert_close_f32};
use rutilelib::blas::BlasBackend;
use rutilelib::blas::GenericBlas;

use rand::Rng;

/// ----------------------
/// Tiled MatMul using Layout's tile iterator
/// ----------------------
fn tiled_matmul<B: BlasBackend>(
    backend: &B,
    a: &TensorView<'_, f32>,
    b: &TensorView<'_, f32>,
    c: &mut TensorViewMut<'_, f32>,
    tiler: &Layout, // The tile layout
    tm : usize,
    tn : usize,
    tk : usize,
) {
    // Iterate over tiles using Layout's iterator
    for (tile_idx) in c.layout().tile_iter(tiler) {
        unsafe {
            let mut c_sub = c.subview_2d_mut(tile_idx[0], tile_idx[1], tm, tn);
            let a_sub = a.subview_2d(tile_idx[0], 0, tm, tk);
            let b_sub = b.subview_2d(tile_idx[1], 0, tk, tn);

            // GEMM on the tile using our BLAS wrapper
            gemm_f32(backend, &a_sub, &b_sub, &mut c_sub);
        }
    }
}

/// ----------------------
/// Example Usage
/// ----------------------
fn main() {
    let m = 64;
    let k = 32;
    let n = 48;
    
    let tm = 16;
    let tn = 16;
    let tk = k;

    let mut rng = rand::thread_rng();
    let mut a_data = vec![0f32; m * k];
    let mut b_data = vec![0f32; k * n];

    for x in a_data.iter_mut() { *x = rng.gen_range(-1.0..1.0); }
    for x in b_data.iter_mut() { *x = rng.gen_range(-1.0..1.0); }

    let shape_a = Shape::new(Tuple::int(vec![m, k]));
    let shape_b = Shape::new(Tuple::int(vec![k, n]));
    let shape_c = Shape::new(Tuple::int(vec![m, n]));

    let a = Tensor::new(a_data, Layout::row_major(shape_a));
    let b = Tensor::new(b_data, Layout::row_major(shape_b));
    let mut c_tiled = Tensor::new(vec![0f32; m * n], Layout::row_major(shape_c.clone()));
    let mut c_ref   = Tensor::new(vec![0f32; m * n], Layout::row_major(shape_c));

    let backend = GenericBlas;

    // Define tile sizes as a Layout
    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![tm, tn])));

    // Run tiled matmul using Layout's tile iterator
    tiled_matmul(&backend, &a.as_view(), &b.as_view(), &mut c_tiled.as_view_mut(), &tiler, tm, tn, tk);

    // Reference GEMM for verification
    gemm_f32(&backend, &a.as_view(), &b.as_view(), &mut c_ref.as_view_mut());

    // Compare results
    let eps = 1e-3;
    unsafe {
        assert_close_f32(m*n, c_tiled.as_view().as_ptr(), c_ref.as_view().as_ptr(), eps);
    }

    println!("Tiled GEMM matches BLAS reference!");
}

