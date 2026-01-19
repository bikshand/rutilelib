use rutilelib::tensor::{Tensor, TensorView, TensorViewMut};
use rutilelib::layout::Layout;
use rutilelib::shape::Shape;
use rutilelib::tuple::Tuple;
use rutilelib::blas::{BlasBackend, GenericBlas};
use rutilelib::gemm::gemm_f32;
use rutilelib::tiled_tensor::TiledTensorViewMut;

use rand::Rng;

fn tiled_gemm<B: BlasBackend>(
    backend: &B,
    a: TensorView<'_, f32>,
    b: TensorView<'_, f32>,
    c: TensorViewMut<'_, f32>,
    tile_m: usize,
    tile_n: usize,
) {
    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![tile_m, tile_n])));

    let mut tiled_c = TiledTensorViewMut::new(c, tiler);

    for (tile, mut c_tile) in tiled_c.tiles_mut() {
        let m0 = tile.start(0);
        let n0 = tile.start(1);
        let tm = tile.len(0);
        let tn = tile.len(1);

        let a_sub = unsafe { a.subview_2d(m0, 0, tm, a.layout().shape().flat_at(1)) };
        let b_sub = unsafe { b.subview_2d(0, n0, b.layout().shape().flat_at(0), tn) };

        gemm_f32(backend, &a_sub, &b_sub, &mut c_tile, 1.0, 0.0);
    }
}

fn main() {
    let (m, k, n) = (64, 32, 48);

    let mut rng = rand::thread_rng();

    let a_data: Vec<f32> = (0..m*k).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_data: Vec<f32> = (0..k*n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let a = Tensor::new(
        a_data,
        Layout::row_major(Shape::new(Tuple::int(vec![m, k]))),
    );

    let b = Tensor::new(
        b_data,
        Layout::row_major(Shape::new(Tuple::int(vec![k, n]))),
    );

    let mut c_tiled = Tensor::new(
        vec![0.0; m*n],
        Layout::row_major(Shape::new(Tuple::int(vec![m, n]))),
    );

    let mut c_ref = Tensor::new(
        vec![0.0; m*n],
        Layout::row_major(Shape::new(Tuple::int(vec![m, n]))),
    );

    let backend = GenericBlas;

    tiled_gemm(&backend, a.as_view(), b.as_view(), c_tiled.as_view_mut(), 16, 16);
    gemm_f32(&backend, &a.as_view(), &b.as_view(), &mut c_ref.as_view_mut(), 1.0, 0.0);

    unsafe {
        rutilelib::gemm::assert_close_f32(
            m * n,
            c_tiled.as_view().as_ptr(),
            c_ref.as_view().as_ptr(),
            1e-3,
        );
    }

    println!("Tiled GEMM matches BLAS reference");
}

