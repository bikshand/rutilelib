# RutileLib

RutileLib is a Rust library for backend-agnostic tiled tensors, inspired by NVIDIA CuTe.

<img width="722" height="580" alt="image" src="https://github.com/user-attachments/assets/fddc6be8-5c0b-4f2c-9884-9dcb98930a5b" />


## Goals

- Provide static and dynamic tensor shapes and indices (`Tuple`, `Shape`).
- Backend-agnostic: not tied to CUDA/NVIDIA.
- Support GETT-style algorithms beyond GEMM.
- Support IM2COL like transformations for CONV.
- Support different memory layouts (e.g. NHWC8).
- Flexible layout algebra for efficient tensor operations.
- Threading and tiling support in a safe, Rust idiomatic way.

## Features

- `Indices`, `Tuple`, and `Shape` for tensor indexing.
- `Layout` algebra for arbitrary memory layouts.
- `TensorView` for views over existing memory.
- Early algorithms for tiled GEMM.
- Designed for future extension to GETT and other tensor operations.

## Usage

```rust
use rutilelib::tuple::Tuple;
use rutilelib::shape::Shape;

let t = Tuple::<3>::new([2, 4, 6]);
let s = Shape::dynamic(vec![2, 4, 6]);

