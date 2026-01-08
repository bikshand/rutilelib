use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::layout::Layout;
use crate::shape::Shape;
use crate::tuple::Tuple;

/* ========================= Tensor ========================= */

pub struct Tensor<T> {
    data: Vec<T>,
    layout: Layout,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, layout: Layout) -> Self {
        assert_eq!(data.len(), layout.size());
        Self { data, layout }
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            ptr: unsafe { NonNull::new_unchecked(self.data.as_ptr() as *mut T) },
            layout: self.layout.clone(),
            _marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            ptr: unsafe { NonNull::new_unchecked(self.data.as_mut_ptr()) },
            layout: self.layout.clone(),
            _marker: PhantomData,
        }
    }
}

/* ========================= TensorView ========================= */

pub struct TensorView<'a, T> {
    pub(crate) ptr: NonNull<T>,
    layout: Layout,
    _marker: PhantomData<&'a T>,
}

pub struct TensorViewMut<'a, T> {
    pub(crate) ptr: NonNull<T>,
    layout: Layout,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> TensorView<'a, T> {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub unsafe fn get(&self, crd: &Tuple) -> &'a T {
        let idx = self.layout.crd2idx(crd);
        &*self.ptr.as_ptr().add(idx)
    }

    /* ---------- N-D subview ---------- */

    pub unsafe fn subview(&self, start: &Tuple, subshape: &Shape) -> TensorView<'a, T> {
        let offset = self.layout.crd2idx(start);

        TensorView {
            ptr: NonNull::new_unchecked(self.ptr.as_ptr().add(offset)),
            layout: Layout::with_shape_stride(
                subshape.clone(),
                self.layout.stride().clone(),
            ),
            _marker: PhantomData,
        }
    }

    /* ---------- 2D fast path ---------- */

    pub unsafe fn subview_2d(
        &self,
        r0: usize,
        c0: usize,
        r: usize,
        c: usize,
    ) -> TensorView<'a, T> {
        let start = Tuple::tup(vec![
            Tuple::int(vec![r0]),
            Tuple::int(vec![c0]),
        ]);

        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![r]),
            Tuple::int(vec![c]),
        ]));

        self.subview(&start, &shape)
    }
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub unsafe fn get_mut(&mut self, crd: &Tuple) -> &'a mut T {
        let idx = self.layout.crd2idx(crd);
        &mut *self.ptr.as_ptr().add(idx)
    }

    pub unsafe fn subview_mut(&mut self, start: &Tuple, subshape: &Shape) -> TensorViewMut<'a, T> {
        let offset = self.layout.crd2idx(start);

        TensorViewMut {
            ptr: NonNull::new_unchecked(self.ptr.as_ptr().add(offset)),
            layout: Layout::with_shape_stride( 
                subshape.clone(),
                self.layout.stride().clone(),
            ),
            _marker: PhantomData,
        }
    }

    pub unsafe fn subview_2d_mut(
        &mut self,
        r0: usize,
        c0: usize,
        r: usize,
        c: usize,
    ) -> TensorViewMut<'a, T> {
        let start = Tuple::tup(vec![
            Tuple::int(vec![r0]),
            Tuple::int(vec![c0]),
        ]);

        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![r]),
            Tuple::int(vec![c]),
        ]));

        self.subview_mut(&start, &shape)
    }
}

/* ========================= Tests ========================= */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Layout, RowMajor};

    #[test]
    fn tensor_create_and_view() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);
        let data = (0..6).collect::<Vec<i32>>();

        let t = Tensor::new(data, layout);
        let v = t.as_view();

        assert_eq!(v.layout().shape().size(), 6);
    }

    #[test]
    fn tensorview_subview_2d() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![4]),
            Tuple::int(vec![4]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);
        let data = (0..16).collect::<Vec<i32>>();

        let t = Tensor::new(data, layout);
        let v = t.as_view();

        let sub = unsafe { v.subview_2d(1, 1, 2, 2) };

        assert_eq!(sub.layout().shape().size(), 4);
        assert_eq!(sub.layout().shape().to_string(), "(2,2)");
    }

    #[test]
    fn tensorview_nd_subview() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
            Tuple::int(vec![4]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);
        let data = (0..24).collect::<Vec<i32>>();

        let t = Tensor::new(data, layout);
        let v = t.as_view();

        let start = Tuple::tup(vec![
            Tuple::int(vec![1]),
            Tuple::int(vec![1]),
            Tuple::int(vec![1]),
        ]);

        let subshape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![1]),
            Tuple::int(vec![2]),
            Tuple::int(vec![3]),
        ]));

        let sub = unsafe { v.subview(&start, &subshape) };
        assert_eq!(sub.layout().shape().size(), 6);
    }

    #[test]
    fn tensorview_mut_write() {
        let shape = Shape::new(Tuple::tup(vec![
            Tuple::int(vec![3]),
            Tuple::int(vec![3]),
        ]));

        let layout = Layout::new::<RowMajor>(shape);
        let mut t = Tensor::new(vec![0i32; 9], layout);

        let mut v = t.as_view_mut();
        let mut sub = unsafe { v.subview_2d_mut(1, 1, 1, 1) };

        unsafe {
            *sub.ptr.as_ptr() = 42;
        }

        let full = t.as_view();
        let val = unsafe {
            full.get(&Tuple::tup(vec![
                Tuple::int(vec![1]),
                Tuple::int(vec![1]),
            ]))
        };

        assert_eq!(*val, 42);
    }
}

