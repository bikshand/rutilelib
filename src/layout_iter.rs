use crate::shape::Shape;
use crate::tuple::Tuple;
use crate::layout::Layout;
use crate::layout_algebra::{flat_divide};

pub struct LayoutIterator {
    shape: Vec<usize>,   // owns tile dimensions
    current: Vec<usize>,
    done: bool,
}

impl LayoutIterator {
    pub fn new(shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        Self {
            shape,
            current: vec![0; ndim],
            done: false,
        }
    }
}

impl Iterator for LayoutIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment index lexicographically
        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
    }
}


#[test]
fn test_tile_iter_2d() {
    // Layout 2D: 8x6
    let layout = Layout::new::<crate::layout::RowMajor>(
        Shape::new(Tuple::int(vec![8, 6]))
    );

    // Tile size: 3x2
    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![3, 2])));

    let mut tiles = Vec::new();
    for idx in layout.tile_iter(&tiler) {
        tiles.push(idx.clone());
    }

    // Each idx should have start positions of the tile in each dim
    // There should be 8/3 = 2 tiles along rows (ignore remainder), 6/2 = 3 tiles along cols
    assert_eq!(tiles.len(), 6); // 2*3
    assert_eq!(tiles[0], vec![0,0]);
    assert_eq!(tiles[1], vec![0,2]);
    assert_eq!(tiles[2], vec![0,4]);
    assert_eq!(tiles[3], vec![3,0]);
    assert_eq!(tiles[4], vec![3,2]);
    assert_eq!(tiles[5], vec![3,4]);
}

#[test]
fn test_rest_iter_2d() {
    // Layout 2D: 8x6
    let layout = Layout::new::<crate::layout::RowMajor>(
        Shape::new(Tuple::int(vec![8, 6]))
    );

    // Tile size: 3x2
    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![3, 2])));

    let mut rests = Vec::new();
    for idx in layout.rest_iter(&tiler) {
        rests.push(idx.clone());
    }

    // Rest iterator should cover remaining indices along rows and cols
    // Remaining row: 8 % 3 = 2, remaining col: 6 % 2 = 0
    // So there is a remainder block of shape 2x6
    assert_eq!(rests.len(), 1); 
    assert_eq!(rests[0], vec![6,0]); // starting index of remainder
}

#[test]
fn test_tile_iter_3d() {
    // 3D layout: 4 x 6 x 8
    let layout = Layout::new::<crate::layout::RowMajor>(
        Shape::new(Tuple::int(vec![4, 6, 8]))
    );

    // Tile size: 2x3x4
    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![2,3,4])));

    let mut tiles = Vec::new();
    for idx in layout.tile_iter(&tiler) {
        tiles.push(idx.clone());
    }

    // Expect 2*2*2 = 8 tiles
    assert_eq!(tiles.len(), 8);
    assert_eq!(tiles[0], vec![0,0,0]);
    assert_eq!(tiles[7], vec![2,3,4]);
}

#[test]
fn test_rest_iter_3d() {
    // 3D layout: 4 x 6 x 8
    let layout = Layout::new::<crate::layout::RowMajor>(
        Shape::new(Tuple::int(vec![4, 6, 8]))
    );

    let tiler = Layout::row_major(Shape::new(Tuple::int(vec![3,4,5])));

    let mut rests = Vec::new();
    for idx in layout.rest_iter(&tiler) {
        rests.push(idx.clone());
    }

    // Remaining sizes along each dim: 4%3=1, 6%4=2, 8%5=3
    assert_eq!(rests.len(), 1); 
    assert_eq!(rests[0], vec![3,4,5]); // starting index of remaining block
}

