use libloading::Library;
use std::sync::OnceLock;

/* ============================================================
   CBLAS ABI (minimal)
   ============================================================ */

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans   = 112,
}

pub type CblasSgemm = unsafe extern "C" fn(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
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

/* ============================================================
   BLAS Backend Trait
   ============================================================ */

#[derive(Copy, Clone, Debug)]
pub enum BlasTranspose {
    NoTrans,
    Trans,
}

pub trait BlasBackend {
    fn gemm_f32(
        &self,
        ta: BlasTranspose,
        tb: BlasTranspose,
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
   Generic BLAS Loader (OpenBLAS / MKL / BLAS)
   ============================================================ */

struct BlasSymbols {
    _lib: Library,
    sgemm: CblasSgemm,
}

static BLAS: OnceLock<BlasSymbols> = OnceLock::new();

fn load_blas() -> &'static BlasSymbols {
    BLAS.get_or_init(|| unsafe {
        let lib = Library::new("libopenblas.so")
            .or_else(|_| Library::new("libblas.so"))
            .expect("Failed to load BLAS library");

        let sgemm = *lib
            .get::<CblasSgemm>(b"cblas_sgemm\0")
            .expect("Failed to load cblas_sgemm");

        BlasSymbols { _lib: lib, sgemm }
    })
}

pub struct GenericBlas;

impl BlasBackend for GenericBlas {
    fn gemm_f32(
        &self,
        ta: BlasTranspose,
        tb: BlasTranspose,
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
    ) {
        let blas = load_blas();

        let transa = match ta {
            BlasTranspose::NoTrans => CBLAS_TRANSPOSE::CblasNoTrans,
            BlasTranspose::Trans   => CBLAS_TRANSPOSE::CblasTrans,
        };

        let transb = match tb {
            BlasTranspose::NoTrans => CBLAS_TRANSPOSE::CblasNoTrans,
            BlasTranspose::Trans   => CBLAS_TRANSPOSE::CblasTrans,
        };

        unsafe {
            (blas.sgemm)(
                CBLAS_LAYOUT::CblasRowMajor,
                transa,
                transb,
                m, n, k,
                alpha,
                a, lda,
                b, ldb,
                beta,
                c, ldc,
            );
        }
    }
}

