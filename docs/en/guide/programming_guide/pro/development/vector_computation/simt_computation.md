# SIMT Computation

<!-- md-trans-meta sourceCommit=49c9842f2975007e96468639469c30745f0bd315 translatedAt=2026-10-27T01:59:21.736Z pushedAt=2026-10-27T07:40:47.208Z nl=2 -->

SIMT computation uses threads as the basic execution unit and is suitable for expressing runtime indexing, per-thread branching, and atomic updates to shared addresses. PyPTO Pro supports hybrid SIMD and SIMT programming: the outer JIT Kernel manages tiles, data movement, and pipeline dependencies, while SIMT functions perform per-thread computation.

For a first run, see [Softmax Operator Quick Start (SIMT)](../../../../quick_start/pro/softmax_simt.md). For concepts, see [SIMT Programming Paradigm](../../programming_paradigm/SIMT/programming_paradigm.md). For complete API constraints, see [SIMT API](../../../../../api/pro_api/SIMT-API/index.md).

## Defining and Launching SIMT Functions

### Defining Entrypoint Functions and Helper Functions

An entrypoint function is defined with @pypto_pro.language.vector_function(mode="simt", max_threads=N). Each thread executes the function body once, and the result is written back through a Tensor or Tile. A helper function is defined with @pypto_pro.language.vector_function(mode="simt"), executes in the current thread, and can return a Scalar.

The following example computes output = input_tensor * scale + bias for 1000 FP32 elements. The helper function reuses the per-thread computation, and the entrypoint function locates elements through the Block and Thread indices and checks the tail boundary.

```python
import pypto_pro.language as pl


ELEMENTS = 1000
THREADS = 256


@pl.vector_function(mode="simt")
def affine(value: pl.DT_FP32, scale: pl.DT_FP32, bias: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + bias


@pl.vector_function(mode="simt", max_threads=THREADS)
def transform(
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    input_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    if index < count:
        output[0, index] = affine(input_tensor[0, index], scale, bias)
```

Parameter types can be inferred from the arguments. Tensor and Scalar parameters can be annotated for readability and type checking, while Tile parameters do not support annotations.

### Calling a Thread Block in the Outer Vector Execution Domain

In the Vector execution domain of the outer Kernel, call the SIMT entrypoint function through `simt_func[threads](...)`.

```python
import pypto_pro.language as pl

@pl.jit(arch="3510")
def transform_kernel(
    input_tensor: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    output: pl.Tensor[[1, ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    with pl.section_vector():
        transform[THREADS](output, input_tensor, count, scale, bias)
```

max_threads declares the upper limit of a single thread block, threads specifies the actual size of this call, and arguments are passed positionally inside the parentheses. The entrypoint function cannot be called in a nested manner within a SIMT function.

### Launching the Outer Kernel from the Host

Launch the outer jit kernel function from the Host:

```python
import torch
import torch_npu


torch.npu.set_device(0)
input_tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="npu:0").reshape(1, ELEMENTS)
output = torch.empty_like(input_tensor)
scale = 2.0
bias = 1.0
blocks = (ELEMENTS + THREADS - 1) // THREADS

transform_kernel[None, blocks](input_tensor, output, ELEMENTS, scale, bias)
torch.npu.synchronize()
torch.testing.assert_close(output, input_tensor * scale + bias, rtol=0, atol=0)
```

In this example, `block_dim` is set to 4 on the Host side to request launching four logical Vector cores. Assuming all four cores actually take effect, each Vector core calls `transform[THREADS](...)` once, each launching a Thread Block containing 256 threads. In the last Thread Block, only 232 threads access data, and the remaining 24 threads are skipped by the boundary check.

The Host launch parameter `block_dim` is used to configure the number of logical cores, while `pypto_pro.language.simt.block_dim()` inside the SIMT function indicates the number of threads of the Thread Block in each dimension. The two have different meanings. For the call form and default values of the Host launch parameters, see [Kernel Function](../kernel_function.md#meaning-and-setting-of-blockdim).

## Configuring Threads and Mapping Data Indices

The threads in square brackets can be written as one to three integer expressions, for example, `simt_func[256](...)`, `simt_func[16, 16](...)`, and `simt_func[8, 8, 4](...)`. pypto_pro.language.simt.block_dim() returns the corresponding three-dimensional size, with unspecified dimensions padded to 1. Each dimension must be a compile-time positive integer, and the product must not exceed max_threads and must not exceed 2048. For example, `simt_func[8, 8, 4](...)` indicates that each thread block has 256 threads.

A one-dimensional thread block can use the global index in the preceding example. A two-dimensional or three-dimensional thread block can be flattened using the X-dimension-first [pypto_pro.language.simt.linear_thread_idx](../../../../../api/pro_api/SIMT-API/execution/linear_thread_idx.md), and then the global index is calculated by combining the thread block number and the total number of threads per block:

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def copy_3d(
    source: pl.Tensor[[1, 1024], pl.DT_FP32],
    output: pl.Tensor[[1, 1024], pl.DT_FP32],
):
    dims = pl.simt.block_dim()
    threads_per_block = dims.x * dims.y * dims.z
    index = pl.simt.block_idx().x * threads_per_block + pl.simt.linear_thread_idx()
    if index < 1024:
        output[0, index] = source[0, index]
```

If the outer Kernel calls this SIMT entrypoint function through `copy_3d[8, 8, 4](...)`, pypto_pro.language.simt.linear_thread_idx() returns a number from 0 to 255 within the block; pypto_pro.language.simt.block_idx().x multiplied by 256 provides the block offset. For example, when there are actually 4 thread blocks, they process indices 0 to 255, 256 to 511, 512 to 767, and 768 to 1023, respectively.

## Accessing Scalar, GM Tensor, and UB Tile

| Object | Access Method | Usage Requirements |
|---|---|---|
| Scalar | Local variables, arithmetic, comparison, Boolean expressions, and the pypto_pro.language.simt scalar APIs | Each thread computes independently; multiple operands of a math function must have the same dtype. |
| GM Tensor | Full scalar subscripts such as tensor[i, j] | Non-zero rank, static shape, ND layout; indices can be computed at runtime, and the program must ensure that the indices are valid. |
| UB Tile | tile[row, col] | Static 2D shape, UB, ND layout; created and managed by the outer kernel. |
| Tile valid region | tile.valid_shape[0], tile.valid_shape[1] | Read the valid number of rows and columns at runtime, and protect subscripts according to the valid range when accessing the tail block. |

Tensor/Tile must be passed as complete objects; elements, slices, or tile subviews are not supported as function parameters. The element bit width must not be less than 8 bits; specific scalar computations and atomic operations must still satisfy their respective dtype cubes. Runtime indices are not equivalent to dynamic tensor shapes, which are currently not supported.

### Accessing the Valid Region of a Tile

The following entrypoint processes a Tile with the shape [2, 128], and the caller uses `add_valid[128, 2](...)`. The thread coordinates cover the physical shape, while the actual read/write range is controlled by valid_shape:

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def add_valid(data, delta: pl.DT_FP32):
    row = pl.simt.thread_idx().y
    col = pl.simt.thread_idx().x
    if row < data.valid_shape[0] and col < data.valid_shape[1]:
        data[row, col] = data[row, col] + delta
```

The valid region is set by the outer Kernel and passed in. SIMT entrypoint function calls do not automatically add boundary checks for Tensor/Tile element reads and writes. SIMT functions also cannot create Tiles or call block-level data movement APIs such as pypto_pro.language.load and pypto_pro.language.store.

### Scalar Computation and Type Conversion

SIMT functions can use common Scalar expressions and also provide the [Scalar Computation and Type Conversion API](../../../../../api/pro_api/SIMT-API/scalar_compute/index.md), which supports the following features:

- Type conversion: pypto_pro.language.simt.cast is used for numeric conversion, and pypto_pro.language.simt.bitcast is used to reinterpret the binary bit pattern.
- Basic math operations: pypto_pro.language.simt.abs, pypto_pro.language.simt.min, pypto_pro.language.simt.max, pypto_pro.language.simt.sqrt, pypto_pro.language.simt.rsqrt, and pypto_pro.language.simt.fma provide absolute value, minimum/maximum, square root, reciprocal square root, and fused multiply-add, respectively.
- Exponential, logarithmic, and trigonometric functions: including pypto_pro.language.simt.exp, pypto_pro.language.simt.exp2, pypto_pro.language.simt.log, pypto_pro.language.simt.log2, pypto_pro.language.simt.log1p, pypto_pro.language.simt.sin, pypto_pro.language.simt.cos, and pypto_pro.language.simt.tanh.
- Rounding operations: pypto_pro.language.simt.rint, pypto_pro.language.simt.round, pypto_pro.language.simt.floor, pypto_pro.language.simt.ceil, and pypto_pro.language.simt.trunc support rounding according to different rules.
- Floating-point value checks: pypto_pro.language.simt.isnan and pypto_pro.language.simt.isinf determine whether a value is NaN or infinite, respectively.

## Handling Data Dependencies

### Outer Pipeline Synchronization

SIMT entrypoint function calls are executed asynchronously on the V pipeline. In mixed computation, data moved in by MTE2 must be ready before it can be accessed by SIMT, and UB data updated by SIMT must be computed before it can be moved out by MTE3:

```text
load (MTE2) -> MTE2/V synchronization -> SIMD or SIMT computation (V) -> V/MTE3 synchronization -> store (MTE3)
```

For ordinary tiles, dependencies are expressed through the paired pypto_pro.language.system.sync_src and pypto_pro.language.system.sync_dst. For the complete code, see [pypto_pro.language.system.sync_src call examples](../../../../../api/pro_api/SIMD-API/synchronization/sync_src.md#examples).

When using pypto_pro.language.make_tile_group with mutex_ids, the Auto Mutex enabled by default can manage the buffer dependencies of pypto_pro.language.load, SIMT entrypoint function calls, and pypto_pro.language.store. Enabling only auto_mutex=True does not automatically complete synchronization for ordinary pypto_pro.language.make_tile.

### Atomic Update

When multiple threads operate on the same element, use the atomic operation API series. Atomic operations do not provide a cross-block barrier, nor do they guarantee that data at other addresses is ready. For the complete type and return value rules, see [Atomic Operation API](../../../../../api/pro_api/SIMT-API/atomic/index.md).

## Example: Implementing Gather with Scalar Indices

The following example reads data from the input Tensor by the row numbers specified in indices:

```text
output[row, col] = input_tensor[indices[0, row], col]
```

This example expresses Gather semantics through Tensor element access. Each thread processes one or more output rows, and a loop is used within a row for copying. This is an example of index expression; actual performance still needs to be measured in combination with data layout and thread mapping.

```python
import pypto_pro.language as pl


INPUT_ROWS = 100000
WIDTH = 128
OUTPUT_ROWS = 12288
THREADS = 256
BLOCKS = (OUTPUT_ROWS + THREADS - 1) // THREADS


@pl.vector_function(mode="simt", max_threads=THREADS)
def gather_rows(
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    row_count: pl.DT_UINT32,
):
    first_row = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.thread_idx().x
    row_stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
    for row in pl.range(first_row, row_count, row_stride):
        input_row = indices[0, row]
        for col in pl.range(0, WIDTH, 1):
            output[row, col] = input_tensor[input_row, col]


@pl.jit(arch="3510")
def gather_kernel(
    input_tensor: pl.Tensor[[INPUT_ROWS, WIDTH], pl.DT_FP32],
    indices: pl.Tensor[[1, OUTPUT_ROWS], pl.DT_INT32],
    output: pl.Tensor[[OUTPUT_ROWS, WIDTH], pl.DT_FP32],
    row_count: pl.DT_UINT32,
):
    with pl.section_vector():
        gather_rows[THREADS](output, input_tensor, indices, row_count)
```

The caller must ensure that 0 <= row_count <= OUTPUT_ROWS, and that all indices elements being read are within [0, INPUT_ROWS). The shapes are fixed, while the index values can change at runtime. After preparing the NPU Tensor that satisfies the annotations, launch it from the Host:

```python
gather_kernel[None, BLOCKS](input_tensor, indices, output, OUTPUT_ROWS)
```

In this example, on the Host side, `block_dim` is set to `BLOCKS=48`, requesting the launch of 48 logical Vector cores. Each actually effective Vector core launches a Thread Block containing 256 threads. Threads process subsequent rows with a stride of `pypto_pro.language.simt.grid_dim().x * pypto_pro.language.simt.block_dim().x`, covering all 12288 rows. Each thread writes to a different output row, and there is no data dependency between rows, so no thread block barrier is required.

## Current Capability Boundaries

- A SIMT entrypoint must be invoked from the outer Vector execution domain. Directly launching a SIMT function from the Host or nesting a SIMT entrypoint function call within a SIMT function is not supported.
- Tile creation, SIMD Tile computation, Reg computation, and System pipeline operations are not supported in SIMT.
- Dynamic GM Shape, Tile Subview, L1 Buffer Tile, DN/NZ layout, and generic pointer parameters are not supported.
- Warp shuffle/vote/reduce, thread-private arrays, and explicit Cached GM access APIs are not provided.
