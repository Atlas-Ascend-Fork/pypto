# Tiling Configuration

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T02:35:55.115Z pushedAt=2026-10-28T08:59:51.043Z nl=2 -->

Properly setting TileShape is critical for optimizing operator performance. TileShape defines how data is split across different computation units of the hardware, affecting data movement and computation efficiency. By setting TileShape appropriately, you can significantly improve computing performance, reduce data movement overhead, and achieve efficient computation.

## Overview

The core of TileShape configuration lies in properly splitting data blocks based on hardware resources and computing requirements, so as to maximize hardware resource utilization, reduce data movement overhead, and thereby improve computing performance.

- Vector computation: In vector computation, `set\_vec\_tile\_shapes` is used to set the split size of vector data in each dimension. Proper splitting allows data to fully utilize the Unified Buffer (UB) and be efficiently processed on the vector computation unit.
- Cube computation: In cube computation, the cube multiplication shape change is denoted as \(m, k\) x \(k, n\) = \(m, n\). `set\_cube\_tile\_shapes` is used to sequentially set the tile size of the cube in the m, k, and n dimensions. Proper splitting can fully utilize the L0 and L1 buffers and reduce data movement overhead.

## Tiling Configuration for Vector Computation

set\_vec\_tile\_shapes is used to set the TileShape of each dimension in vector computation.

```python
# Set the TileShape for vector computation
pypto.set_vec_tile_shapes(1, 1, 8, 8)
# Obtain and print the configured TileShape
print(pypto.get_vec_tile_shapes())  # Output: [1, 1, 8, 8]
```

`pypto.set\_vec\_tile\_shapes\(1, 1, 8, 8\)` indicates that the vector has four dimensions, each dimension is split by sizes of 1, 1, 8, and 8, respectively, and the original vector is transferred to the UB for computation based on the split size of \(1, 1, 8, 8\).

The following is an actual use case:

```python
@pypto.frontend.jit
def compute_with_vec_tile_shapes_kernel(
    a: pypto.Tensor((32, 32), pypto.DT_FP32),
    b: pypto.Tensor((32, 32), pypto.DT_FP32),
    out: pypto.Tensor((32, 32), pypto.DT_FP32),
    set_shapes: tuple
):
    pypto.set_vec_tile_shapes(*set_shapes)
    out[:] = pypto.add(a, b)

def compute_with_vec_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: tuple, dynamic: bool = False) -> torch.Tensor:
    # Call by directly passing a torch tensor
    out = torch.empty_like(a)
    compute_with_vec_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_vec_tile_shapes_basic():
    ...
    a = torch.tensor([[[1, 2, 3],
                       [1, 2, 3]]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[[4, 5, 6],
                       [4, 5, 6]]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[[5, 7, 9],
                            [5, 7, 9]]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = (1, 2, 8)
    out = compute_with_vec_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

The preceding use case demonstrates the use of `set_vec_tile_shapes` in a simple vector addition scenario.

It should be noted that different TileShape settings generally do not affect the vector computation result, but they do affect the execution time of the vector computation, as shown in the following use case:

```python
@pypto.frontend.jit
def compute_with_vec_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 4, 128)
    out[:] = pypto.add(a, b)

@pypto.frontend.jit
def compute_with_vec_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 4, 8, 256)
    out[:] = pypto.add(a, b)

def compute_with_vec_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_vec_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_vec_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_vec_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_vec_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_vec_tile_shapes(1, 2, 4, 128)): {runtime_1}")
    print(f"runtime_2(pypto.set_vec_tile_shapes(2, 4, 8, 256)): {runtime_2}")
```

In this example, for the addition of two vectors with the shape \(4, 32, 64, 256\), the execution time of set\_vec\_tile\_shapes\(1, 2, 4, 128\) is significantly longer than that of set\_vec\_tile\_shapes\(2, 4, 8, 256\).

For the complete example, see [tiling_config.py](../../../../../../examples/01_beginner/tiling/tiling_config.py).

## Tiling Configuration for Cube Computation

`set\_cube\_tile\_shapes` is used to set the TileShape of each cube in the m, k, and n dimensions during cube computation. For specific optimization tiling configurations, see [Matmul High-Performance Programming](../debug/matmul_performance_guide.md).

```python
# Set the TileShape for Cube computation
pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], enable_split_k=False)
# Obtain and print the configured TileShape
print(pypto.get_cube_tile_shapes())  # Output: [[16, 16], [256, 512], [128, 128], False]
```

pypto.set\_cube\_tile\_shapes\(\[16, 16\], \[256, 512\], \[128, 128\], enable\_split\_k=False\): the shape change of cube multiplication is denoted as \(m, k\) x \(k, n\) = \(m, n\). Here, the three lists set the tile sizes of the m, k, and n dimensions of the cube respectively. For each list, the first element sets the tile size of L0, and the second element sets the tile size of L1. The enable\_split\_k parameter specifies whether to enable multi-core K splitting, and the default value is False. For scenarios where M and N are small but the K axis is large, splitting cores only on the M and N axes may fail to fully utilize all cores, resulting in poor overall performance. In this case, you can set `enable_split_k` to True to enable K-axis core splitting.

The following is an actual use case:

```python
@pypto.frontend.jit
def compute_with_cube_tile_shapes_kernel(
    a: pypto.Tensor((64, 64), pypto.DT_FP32),
    b: pypto.Tensor((64, 64), pypto.DT_FP32),
    out: pypto.Tensor((64, 64), pypto.DT_FP32),
    set_shapes: list
):
    pypto.set_cube_tile_shapes(*set_shapes)
    out[:] = pypto.matmul(a, b, a.dtype)

def compute_with_cube_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: list, dynamic: bool = False) -> torch.Tensor:
    # Call by directly passing a torch tensor
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    compute_with_cube_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_cube_tile_shapes_basic():
    ...
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = [[32, 32], [64, 64], [64, 64]]
    out = compute_with_cube_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

The preceding example demonstrates an application of `set\_cube\_tile\_shapes` in a simple cube multiplication scenario.

It should be noted that, in general, setting different TileShapes does not affect the computation result of the cube, but it does affect the execution time of the cube computation, as shown in the following example:

```python
import pypto
import torch
import time

@pypto.frontend.jit
def compute_with_cube_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

@pypto.frontend.jit
def compute_with_cube_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

def compute_with_cube_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_cube_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_cube_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 64, 512), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 128, 512), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_cube_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_cube_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])): {runtime_1}")
    print(f"runtime_2(pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])): {runtime_2}")
```

In this example, for multiplying two cubes with shapes \(4, 64, 512\) and \(4, 512, 128\), the execution time of set\_cube\_tile\_shapes\(\[32, 32\], \[32, 32\], \[32, 32\]\) is significantly longer than that of set\_cube\_tile\_shapes\(\[64, 64\], \[128, 128\], \[128, 128\]\).

For the complete example, see [tiling_config.py](../../../../../../examples/01_beginner/tiling/tiling_config.py).

## Constraints

- When setting TileShape parameters, you must meet the constraints. The values should match the number of dimensions and size of the tensor shape to be processed, and cannot be too small or too large.

    The TileShape value cannot be too small. An excessively small TileShape leads to an excessive number of splits, TensorShape/TileShape (i.e., the product of the per-dimension ratios of TensorShape to TileShape), which in turn causes an excessive number of online loop unrolling iterations. This may cause expression table compilation to fail and increase runtime overhead. The size of the expression table is related to the number of online loop unrolling iterations and the number of operator inputs. It is recommended to keep \(TensorShape/TileShape\)\*\(1+the number of operator inputs\) below 18000.

    The TileShape value cannot be excessive. An excessive TileShape exceeds the storage capacity of the corresponding hardware (buffer). Ensure that the size of the split data (the product of the data type size and the size of each dimension of the split data) does not exceed the storage capacity of the corresponding hardware unit.

    In addition, the number of dimensions for set\_vec\_tile\_shapes must not exceed 5, and the split size of the last axis must be 32B-aligned. For set\_cube\_tile\_shapes, kL0, kL1, nL0, and nL1 must all be 32-byte aligned. For detailed configuration requirements, see the related API documentation.

- Setting TileShape affects the on-board execution time. Generally, the more fully the capacity of hardware units is utilized, that is, the larger the amount of data processed in a single computation, the shorter the execution time. However, a larger TileShape parameter does not necessarily mean faster on-board execution, as the overhead of data movement and other stages must also be considered.

## Other Operations

- Performance observation: You can use performance analysis tools (such as swimlane diagrams) to observe the performance under different TileShape settings, thereby evaluating the rationality of the TileShape configuration and obtaining the optimal TileShape for the current scenario.
- Precision impact: Unless extreme values are set, TileShape generally does not affect precision (which is an undesirable behavior from the framework's perspective).
