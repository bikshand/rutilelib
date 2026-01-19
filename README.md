# RutileLib

RutileLib is a Rust library for backend-agnostic tiled tensors, inspired by NVIDIA CuTe.

Matrix Size: 4K x 4K X 256. Tile Size : 512 X 512
<img width="715" height="570" alt="image" src="https://github.com/user-attachments/assets/f0472e2e-6610-4d1a-bd51-2c82bca4263b" />


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

