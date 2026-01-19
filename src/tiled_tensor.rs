// ============================================================
// tiled_tensor_view.rs
// ============================================================
//
// Core idea:
//   TiledTensorView = TensorView + explicit tile iteration structure
//
// This makes tiling SEMANTIC (not just an optimization), which is
// critical for correctness, aliasing, and autodiff.
//
// ============================================================

use crate::tensor::{TensorView, TensorViewMut};
use crate::layout::Layout;
use crate::layout_algebra::flat_divide;
use crate::tuple::Tuple;
use crate::shape::Shape;

/* ============================================================
   Tile descriptor
   ============================================================ */

#[derive(Debug, Clone)]
pub struct Tile {
    start: Vec<usize>,
    len:   Vec<usize>,
}

impl Tile {
    pub fn start(&self, dim: usize) -> usize {
        self.start[dim]
    }

    pub fn len(&self, dim: usize) -> usize {
        self.len[dim]
    }

    pub fn ndim(&self) -> usize {
        self.start.len()
    }
}

/* ============================================================
   Tile iterator
   ============================================================ */

pub struct TileIter {
    tile_shape: Vec<usize>,
    full_shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl TileIter {
    pub fn new(tile_shape: Vec<usize>, full_shape: Vec<usize>) -> Self {
        let ndim = tile_shape.len();
        Self {
            tile_shape,
            full_shape,
            current: vec![0; ndim],
            done: false,
        }
    }
}

impl Iterator for TileIter {
    type Item = Tile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let start = self.current.clone();
        let mut len = Vec::with_capacity(start.len());

        for d in 0..start.len() {
            let remaining = self.full_shape[d] - start[d];
            len.push(self.tile_shape[d].min(remaining));
        }

        // advance multi-dimensional counter
        for d in (0..self.current.len()).rev() {
            self.current[d] += self.tile_shape[d];
            if self.current[d] < self.full_shape[d] {
                break;
            } else {
                self.current[d] = 0;
                if d == 0 {
                    self.done = true;
                }
            }
        }

        Some(Tile { start, len })
    }
}

/* ============================================================
   TiledTensorView (immutable)
   ============================================================ */

pub struct TiledTensorView<'a, T> {
    base: TensorView<'a, T>,
    tile_layout: Layout,
    tile_iter: TileIter,
}

impl<'a, T> TiledTensorView<'a, T> {
    pub fn new(base: TensorView<'a, T>, tiler: Layout) -> Self {
        // Use layout algebra to compute iteration space
        let flat = flat_divide(base.layout(), &tiler);

        let (tile_dims, full_dims) = match &flat.shape().dims {
            Tuple::Int(v) => {
                let t = tiler.shape().flat_len();
                (v[..t].to_vec(), v[t..].to_vec())
            }
            _ => panic!("flat_divide must produce flat Int tuple"),
        };

        let tile_iter = TileIter::new(tile_dims.clone(), base.layout().shape().dims.flatten());

        Self {
            base,
            tile_layout: tiler,
            tile_iter,
        }
    }

    pub fn tiles(&mut self) -> impl Iterator<Item = (Tile, TensorView<'a, T>)> + '_ {
        self.tile_iter.by_ref().map(|tile| {
            let sub = unsafe {
                self.base.subview(&Tuple::int(tile.start.clone()), &Shape::new(Tuple::int(tile.len.clone())))
            };
            (tile, sub)
        })
    }
}

/* ============================================================
   TiledTensorViewMut (mutable)
   ============================================================ */

pub struct TiledTensorViewMut<'a, T> {
    base: TensorViewMut<'a, T>,
    tile_layout: Layout,
    tile_iter: TileIter,
}

impl<'a, T> TiledTensorViewMut<'a, T> {
    pub fn new(base: TensorViewMut<'a, T>, tiler: Layout) -> Self {
        let flat = flat_divide(base.layout(), &tiler);

        let (tile_dims, rest_dims) = match &flat.shape().dims {
            Tuple::Int(v) => {
                let t = tiler.shape().flat_len();
                (v[..t].to_vec(), v[t..].to_vec())
            }
            _ => panic!("flat_divide must produce flat Int tuple"),
        };

        let tile_iter = TileIter::new(tile_dims.clone(), base.layout().shape().dims.flatten());

        Self {
            base,
            tile_layout: tiler,
            tile_iter,
        }
    }

    pub fn tiles_mut(&mut self) -> impl Iterator<Item = (Tile, TensorViewMut<'a, T>)> + '_ {
        self.tile_iter.by_ref().map(|tile| {
            let sub = unsafe {
                self.base.subview_mut(&Tuple::int(tile.start.clone()), &Shape::new(Tuple::int(tile.len.clone())))
            };
            (tile, sub)
        })
    }
}

/* ============================================================
   Why this matters (design note)
   ============================================================ */

// - Tiles are DISJOINT by construction
// - Iteration structure is explicit
// - Reduction boundaries are visible
// - Backward passes can reuse the same tiling
// - Rust aliasing rules are satisfied structurally
//
// This is the missing abstraction for:
//   - high-performance autodiff
//   - safe scatter/gather
//   - zero-cost tiling
//
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::layout::Layout;
    use crate::shape::Shape;
    use crate::tuple::Tuple;

    fn make_tensor_2d(m: usize, n: usize) -> Tensor<f32> {
        let data: Vec<f32> = (0..m*n).map(|x| x as f32).collect();
        let shape = Shape::new(Tuple::int(vec![m, n]));
        Tensor::new(data, Layout::row_major(shape))
    }

    #[test]
    fn tile_iter_covers_entire_tensor() {
        let m = 8;
        let n = 6;
        let tile = vec![3, 4];

        let t = make_tensor_2d(m, n);
        let tiler = Layout::row_major(Shape::new(Tuple::int(tile.clone())));

        let mut tiled = TiledTensorView::new(t.as_view(), tiler);

        let mut visited = vec![false; m * n];

        for (tile_desc, tile_view) in tiled.tiles() {
            for i in 0..tile_desc.len(0) {
                for j in 0..tile_desc.len(1) {
                    let gi = tile_desc.start(0) + i;
                    let gj = tile_desc.start(1) + j;
                    let idx = gi * n + gj;

                    assert!(!visited[idx], "Element visited twice");
                    visited[idx] = true;

                    let v = unsafe { *tile_view.ptr_at(&Tuple::int(vec![i, j])) };
                    assert_eq!(v, (idx as f32));
                }
            }
        }

        assert!(
            visited.iter().all(|x| *x),
            "Not all elements were covered by tiles"
        );
    }

    #[test]
    fn tile_shapes_are_correct_at_edges() {
        let m = 7;
        let n = 5;
        let tile = vec![4, 3];

        let t = make_tensor_2d(m, n);
        let tiler = Layout::row_major(Shape::new(Tuple::int(tile.clone())));

        let mut tiled = TiledTensorView::new(t.as_view(), tiler);

        for (tile_desc, _) in tiled.tiles() {
            let tm = tile_desc.len(0);
            let tn = tile_desc.len(1);

            assert!(tm <= tile[0]);
            assert!(tn <= tile[1]);

            // Edge tiles must not exceed bounds
            assert!(tile_desc.start(0) + tm <= m);
            assert!(tile_desc.start(1) + tn <= n);
        }
    }

    #[test]
    fn tiled_mutation_updates_base_tensor() {
        let m = 6;
        let n = 6;
        let tile = vec![2, 3];

        let mut t = make_tensor_2d(m, n);
        let tiler = Layout::row_major(Shape::new(Tuple::int(tile.clone())));

        {
            let mut tiled = TiledTensorViewMut::new(t.as_view_mut(), tiler);

            for (tile, mut tile_view) in tiled.tiles_mut() {
                let tile_offset = tile.start[0] * n + tile.start[1];
                for i in 0..tile.len[0] {
                    for j in 0..tile.len[1] {
                        unsafe {
                            *tile_view.ptr_at_mut(&Tuple::int(vec![i, j])) = (tile_offset + (i * n + j)) as f32;
                        }
                    }
                }
            }
        }

        // Check base tensor updated
        for (i, v) in t.data().iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    }

    #[test]
    fn single_tile_equals_whole_tensor() {
        let m = 4;
        let n = 4;

        let t = make_tensor_2d(m, n);
        let tiler = Layout::row_major(Shape::new(Tuple::int(vec![m, n])));

        let mut tiled = TiledTensorView::new(t.as_view(), tiler);

        let tiles: Vec<_> = tiled.tiles().collect();
        assert_eq!(tiles.len(), 1);

        let (tile, view) = &tiles[0];
        assert_eq!(tile.start, vec![0, 0]);
        assert_eq!(tile.len, vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let v = unsafe { *view.ptr_at(&Tuple::int(vec![i,j])) };
                assert_eq!(v, idx as f32);
            }
        }
    }
}


