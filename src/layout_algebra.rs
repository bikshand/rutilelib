// src/layout_algebra.rs
use crate::layout::{Layout, LayoutPolicy, RowMajor};
use crate::shape::Shape;
use crate::tuple::Tuple;

/// ---------- Helper functions for tuple arithmetic ----------
fn subtract_tuples(a: &Tuple, b: &Tuple) -> Tuple {
    match (a, b) {
        (Tuple::Int(av), Tuple::Int(bv)) => {
            assert_eq!(av.len(), bv.len(), "Tuple length mismatch in subtract_tuples");
            Tuple::Int(av.iter().zip(bv.iter()).map(|(x, y)| x - y).collect())
        }
        (Tuple::Tup(av), Tuple::Tup(bv)) => {
            assert_eq!(av.len(), bv.len(), "Tuple length mismatch in subtract_tuples");
            Tuple::Tup(av.iter().zip(bv.iter()).map(|(x, y)| subtract_tuples(x, y)).collect())
        }
        _ => panic!("Tuple shape mismatch in subtract_tuples"),
    }
}

/// Take the first N elements from Tuple recursively
fn take_tuple(a: &Tuple, n: usize) -> Tuple {
    match a {
        Tuple::Int(v) => Tuple::Int(v.iter().take(n).cloned().collect()),
        Tuple::Tup(v) => Tuple::Tup(v.iter().take(n).map(|x| x.clone()).collect()),
    }
}

/// Skip first N elements from Tuple recursively
fn skip_tuple(a: &Tuple, n: usize) -> Tuple {
    match a {
        Tuple::Int(v) => Tuple::Int(v.iter().skip(n).cloned().collect()),
        Tuple::Tup(v) => Tuple::Tup(v.iter().skip(n).map(|x| x.clone()).collect()),
    }
}

/// ---------- Layout Algebra Operations ----------

/// logical_divide: ((TileM,RestM),(TileN,RestN),...)
pub fn logical_divide(layout: &Layout, tiler: &Layout) -> Layout {
    fn recur(l: &Tuple, t: &Tuple) -> Tuple {
        match (l, t) {
            (Tuple::Int(ldims), Tuple::Int(tdims)) => {
                let mut out = Vec::new();
                for (l_i, t_i) in ldims.iter().zip(tdims.iter()) {
                    out.push(Tuple::Int(vec![*t_i, l_i - t_i]));
                }
                Tuple::Tup(out)
            }
            (Tuple::Tup(lv), Tuple::Tup(tv)) => {
                Tuple::Tup(lv.iter().zip(tv.iter()).map(|(l_, t_)| recur(l_, t_)).collect())
            }
            _ => panic!("Tuple shape mismatch in logical_divide"),
        }
    }

    let shape = Shape::new(recur(&layout.shape().dims, &tiler.shape().dims));
    Layout::new::<RowMajor>(shape)
}

/// zipped_divide: ((TileM,TileN),(RestM,RestN,...))
pub fn zipped_divide(layout: &Layout, tiler: &Layout) -> Layout {
    fn recur(l: &Tuple, t: &Tuple) -> (Tuple, Tuple) {
        match (l, t) {
            (Tuple::Int(ldims), Tuple::Int(tdims)) => {
                let tile = Tuple::Int(tdims.clone());
                let rest = Tuple::Int(ldims.iter().zip(tdims.iter()).map(|(l_i, t_i)| l_i / t_i).collect());
                (tile, rest)
            }
            (Tuple::Tup(lv), Tuple::Tup(tv)) => {
                let tiles: Vec<Tuple> = lv.iter().zip(tv.iter()).map(|(l_, t_)| recur(l_, t_).0).collect();
                let rests: Vec<Tuple> = lv.iter().zip(tv.iter()).map(|(l_, t_)| recur(l_, t_).1).collect();
                (Tuple::Tup(tiles), Tuple::Tup(rests))
            }
            _ => panic!("Tuple shape mismatch in zipped_divide"),
        }
    }

    let (tile, rest) = recur(&layout.shape().dims, &tiler.shape().dims);
    let shape = Shape::new(Tuple::Tup(vec![tile, rest]));
    Layout::new::<RowMajor>(shape)
}

/// tiled_divide: ((TileM,TileN), RestM, RestN, ...)
pub fn tiled_divide(layout: &Layout, tiler: &Layout) -> Layout {
    let zipped = zipped_divide(layout, tiler);
    match &zipped.shape().dims {
        Tuple::Tup(v) => {
            let mut flattened = Vec::new();
            // First element is tile tuple
            flattened.push(v[0].clone());
            // Rest elements: flatten rest tuple if Tup
            match &v[1] {
                Tuple::Tup(rest) => flattened.extend(rest.clone()),
                Tuple::Int(rest) => flattened.push(Tuple::Int(rest.clone())),
            }
            Layout::new::<RowMajor>(Shape::new(Tuple::Tup(flattened)))
        }
        _ => panic!("Unexpected shape in tiled_divide"),
    }
}

/// flat_divide: (TileM,TileN, RestM, RestN, ...)
pub fn flat_divide(layout: &Layout, tiler: &Layout) -> Layout {
    let tiled = tiled_divide(layout, tiler);
    fn flatten(t: &Tuple) -> Vec<usize> {
        match t {
            Tuple::Int(v) => v.clone(),
            Tuple::Tup(vs) => vs.iter().flat_map(|x| flatten(x)).collect(),
        }
    }

    let flat_dims = flatten(&tiled.shape().dims);
    Layout::new::<RowMajor>(Shape::new(Tuple::Int(flat_dims)))
}

/// ---------- Unit Tests ----------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::Tuple;
    use crate::layout::Layout;
    use crate::shape::Shape;

    #[test]
    fn test_logical_divide() {
        let layout = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![8]),
            Tuple::int(vec![6]),
        ])));
        let tiler = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ])));

        let result = logical_divide(&layout, &tiler);
        assert_eq!(result.shape().to_string(), "((2,6),(3,3))");
    }

    #[test]
    fn test_zipped_divide() {
        let layout = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![8]),
            Tuple::int(vec![6]),
        ])));
        let tiler = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ])));

        let result = zipped_divide(&layout, &tiler);
        assert_eq!(result.shape().to_string(), "((2,3),(4,2))");
    }

    #[test]
    fn test_tiled_divide() {
        let layout = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![8]),
            Tuple::int(vec![6]),
        ])));
        let tiler = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ])));

        let result = tiled_divide(&layout, &tiler);
        assert_eq!(result.shape().to_string(), "((2,3),4,2)");
    }

    #[test]
    fn test_flat_divide() {
        let layout = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![8]),
            Tuple::int(vec![6]),
        ])));
        let tiler = Layout::new::<RowMajor>(Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ])));

        let result = flat_divide(&layout, &tiler);
        assert_eq!(result.shape().to_string(), "(2,3,4,2)");
    }
}

