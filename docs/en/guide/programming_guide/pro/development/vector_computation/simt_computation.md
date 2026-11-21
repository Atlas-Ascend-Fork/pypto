# SIMT Computation

<!-- md-trans-meta sourceCommit=af1093db00251286d680306e14abab5735b4f90f translatedAt=2026-11-15T02:30:34.198Z pushedAt=2026-11-18T06:03:01.373Z nl=2 -->

SIMT computation uses threads as the basic execution unit, and is suitable for expressing runtime indexing, per-thread branching, and atomic updates of shared addresses.

PyPTO Pro supports launching SIMT kernels directly from the Host, and also supports calling SIMT vector functions (SIMT VF for short) in an outer kernel to combine SIMD and SIMT computation. For the relationship among the thread model, data objects, and the two programming approaches, see [SIMT Programming Paradigm](../../programming_paradigm/SIMT/programming_paradigm.md). For complete API constraints, see [SIMT API](../../../../../api/pro_api/SIMT-API/index.md).

## SIMT Programming

### Defining a SIMT Kernel

A SIMT kernel is defined with @pypto_pro.language.jit(mode="simt", max_threads=N), which directly describes the function body executed by each thread without wrapping it in a Vector execution domain.

max_threads declares the upper limit of the number of threads per thread block. It must be an integer in the range [1, 2048] and does not accept bool. The actual thread configuration is passed in when the Host launches the kernel, and the product of the number of threads in each dimension must not exceed max_threads. This parameter also participates in compile-time register resource allocation and should be selected based on the computational complexity of a single thread. See [Configuring the Maximum Number of Threads](../../programming_paradigm/SIMT/programming_paradigm.md#configuring-the-maximum-number-of-threads).

The definition of a SIMT kernel must satisfy the following constraints:

- Kernel parameters must provide type annotations and accept only Scalar and Tensor.
- The decorator parameters do not support pipeline, tiling_key, datatype specialization, or sanitizer=True. SIMT kernels do not use Auto Mutex; inter-thread synchronization is organized by the developer through the SIMT synchronization APIs.

### Launching a SIMT Kernel from the Host

The Host launches a SIMT kernel in the following two forms:

- `kernel[grid, block](...)`: uses the current NPU execution stream.
- `kernel[stream, grid, block](...)`: specifies the execution stream. When stream is None, the current execution stream is used.

grid specifies the number of thread blocks, and block specifies the number of threads per thread block. A SIMT kernel must specify grid and block through square brackets. Launching directly with `kernel(...)` is not supported.

| Parameter | Configuration Form | Constraint |
|---|---|---|
| grid | A positive integer, or a tuple/list containing one to three positive integers | The product of each dimension does not exceed 65535. |
| block | A positive integer, or a tuple/list containing one to three positive integers | The product of each dimension does not exceed the max_threads specified at definition. |

Components not given in each dimension are padded with 1, and bool is not accepted. For example, `kernel[4, 256](...)` launches 4 thread blocks, each with 256 threads; `kernel[(2, 3, 4), (8, 4, 2)](...)` launches 24 thread blocks, each with 64 threads. grid and block belong to the configuration of this launch. Changing them does not generate a new compiled instance separately.

The definition and call are illustrated as follows, with the per-thread computation logic omitted. data is an FP32 NPU Tensor with Shape [1, 1024] that has been prepared on the Host side:

```python
import pypto_pro.language as pl


@pl.jit(mode="simt", max_threads=256)
def simt_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    ...


simt_kernel[4, 256](data)
```

## SIMD and SIMT Mixed Programming

### Defining a SIMT VF

A SIMT VF is defined using @pypto_pro.language.vector_function(mode="simt", max_threads=N). Each thread executes the function body once, and the result is written back through the passed-in Tensor or Tile.

The constraint on max_threads is consistent with that of [SIMT kernel](#defining-a-simt-kernel). The actual number of threads is specified at the SIMT VF call site.

- A SIMT VF can accept Scalar, Tensor, and Tile as input parameters, and the parameter type can be inferred from the arguments. Tensor and Scalar parameters can be annotated for readability and type checking. Tile parameters do not support annotations.

### Calling SIMT VF

The outer kernel is defined with @pypto_pro.language.jit() and calls the SIMT VF through `simt_func[threads](...)` in the pypto_pro.language.section_vector() execution domain. The thread block size for this call is specified in the square brackets, and arguments are passed by position in the parentheses.

| Thread Block Configuration | Call Form | Threads per Block |
|---|---|---|
| 1D | `simt_func[256](...)` | 256 |
| 2D | `simt_func[16, 16](...)` | 256 |
| 3D | `simt_func[8, 8, 4](...)` | 256 |

Each dimension must be a compile-time positive integer expression. Unspecified components are padded with 1, and the product of each dimension must not exceed the max_threads defined for the SIMT VF. Using a number of threads that is a multiple of 32 can reduce idle thread lanes in the last Warp, but it is not mandatory.

Each Vector core that reaches the call site launches one thread block. The Grid is determined by the number of Vector cores actually in effect, using only the X dimension, with the Y and Z dimensions both set to 1. There is no need to configure the grid separately at the SIMT VF call site.

The SIMT VF executes asynchronously in the V pipeline, and its dependency on data movement must be organized by the outer kernel. See [Outer Pipeline Synchronization](#outer-pipeline-synchronization).

### Launching the Outer Kernel from the Host

The Host launches the outer kernel through `kernel[block_dim](...)` or `kernel[stream, block_dim](...)`, and the SIMT VF is launched when the outer kernel reaches the call site during execution.

The Host launch parameter block_dim is used to configure the number of logical cores, while pypto_pro.language.simt.block_dim() inside the SIMT function indicates the number of threads of the thread block in each dimension. The two have different meanings. For the default values of the Host launch parameters and the number of cores actually in effect, see [Kernel Function](../kernel_function.md#meaning-and-setting-of-blockdim).

The definition and call are illustrated as follows, with the per-thread computation logic omitted. data is an FP32 NPU Tensor of shape [1, 1024] prepared on the Host side:

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=256)
def simt_vf(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    ...


@pl.jit(arch="3510")
def mixed_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    with pl.section_vector():
        simt_vf[256](data)


mixed_kernel[None, 4](data)
```

## SIMT helper

A SIMT helper is used to reuse per-thread computation logic. It is defined with @pypto_pro.language.vector_function(mode="simt") without specifying max_threads. The function body can use Scalar expressions and SIMT APIs.

- A SIMT kernel, SIMT VF, or another SIMT helper uses the helper through a normal function call, without using square brackets to configure threads. The helper executes in the current thread of the caller, inherits the thread configuration of the entry, and does not launch a new thread block.
- A helper can return no value or return a Scalar, and can also write the result to the passed-in Tensor or Tile. The returned Scalar belongs to the current calling thread.
- The parameter constraints of a helper are determined by the calling entry: the SIMT kernel call chain accepts only Scalar and Tensor; the SIMT VF call chain can also accept a UB Tile that meets the requirements.

The following example calls a helper in a SIMT kernel. The call in a SIMT VF is written the same way. data is an FP32 NPU Tensor with the shape [1, 1024] that is ready on the Host side:

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt")
def add_one(value: pl.DT_FP32) -> pl.DT_FP32:
    return value + 1.0


@pl.jit(mode="simt", max_threads=256)
def update_kernel(data: pl.Tensor[[1, 1024], pl.DT_FP32]):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.linear_thread_idx()
    if index < 1024:
        data[0, index] = add_one(data[0, index])


update_kernel[4, 256](data)
```

## Data Object

### Data Objects and Memory Hierarchy

Data objects in SIMT include Scalar, Tensor, and Tile, which are used to express thread-private data, GM data, and UB data, respectively. For the sharing scope, lifecycle, and resource partitioning of memory, see [Memory Hierarchy](../../programming_paradigm/SIMT/programming_paradigm.md#memory-hierarchy).

| Object | Memory Mapping | Access Method | Usage Requirements |
|---|---|---|---|
| Scalar | Thread-private, usually using registers, and possibly stack space | Local variables, arithmetic, comparison, Boolean expressions, and the pypto_pro.language.simt scalar APIs | Each thread computes independently; the data types of the operands must satisfy the constraints of the corresponding APIs. |
| Tensor | GM, global memory | Full scalar subscripts such as tensor[i, j] | Non-zero rank, static shape, ND layout; indices can be computed at runtime, and the program must ensure that the indices are valid. |
| Tile | UB, shared memory | tile[row, col] | Used only in mixed programming, with static 2D shape, UB, ND layout; created and managed by the outer kernel. |

Tensor and Tile elements in SIMT support DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32. Specific Scalar computation and atomic operations must also satisfy the data type constraints of each API. When Tensor and Tile are used as function parameters, they must be passed as complete objects; Slice or Tile Subview is not supported. The Scalar obtained by reading an element can be used as a Scalar parameter.

The shape resolved from a Tensor must be static; the pypto_pro.language.DYNAMIC dimension is not supported. A dimension specialized with pypto_pro.language.STATIC must be resolved to a static value at launch. A runtime index is not equivalent to a dynamic Tensor shape.

A SIMT kernel and the SIMT helpers it calls accept only Scalar and Tensor. A SIMT VF and the SIMT helpers it calls can also accept UB Tile, but L1 Buffer Tile or DN/NZ layout is not supported. Thread-private arrays and an explicit Cached GM access API are not currently provided.

### Tile and Tensor Read/Write

SIMT functions access Tile or Tensor elements through complete scalar subscripts. Take the two-dimensional object a as an example:

| Operation | Expression | Description |
|---|---|---|
| Read | value = a[row, col] | Reads one element, and the result is a Scalar. |
| Write | a[row, col] = value | Writes a Scalar to one element. |
| Read then write back | a[row, col] = a[row, col] + delta | Reads, computes, and then writes back; not atomic. |

### Valid Region for Tensor and Tile Read/Write

Currently, SIMT only checks whether compile-time constant subscripts are out of bounds when reading and writing Tensor and Tile elements; it does not automatically generate runtime boundary checks.

- Static constant subscript: A subscript that the frontend can evaluate to an integer constant is checked against the Shape of the corresponding dimension. A negative subscript or one that exceeds the range of that dimension causes an error. For example, when the Shape is [1, 64], a[0, 64], a[1, 0], and a[0, -1] all cause errors.
- Dynamic subscript: A subscript derived from a runtime Scalar input parameter, a Tensor element, a thread index, or a computation expression of these. The frontend does not infer its value range. The developer must ensure the validity of the data being read or written, and can explicitly add a condition check of 0 <= index < Shape[axis] before the read or write.

To access valid data of a Tile, you must also ensure that the subscript is less than the valid_shape of the corresponding dimension. The valid region is set and passed in by the outer kernel. SIMT does not automatically restrict reads and writes based on valid_shape. Out-of-bounds reads and writes may corrupt adjacent data or cause execution exceptions.

The following entry processes a Tile input parameter data with a shape of [2, 128]. It obtains the valid number of rows and valid number of columns of the Tile at runtime through data.valid_shape[0] and data.valid_shape[1], respectively, and restricts the actual reads and writes to the valid region through condition checks:

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def add_valid(data, delta: pl.DT_FP32):
    row = pl.simt.thread_idx().y
    col = pl.simt.thread_idx().x
    if row < data.valid_shape[0] and col < data.valid_shape[1]:
        data[row, col] = data[row, col] + delta
```

## SIMT Programming API System

SIMT provides the following APIs for thread index query, synchronization, and per-thread computation:

### Thread Hierarchy and Index

The [execution model APIs](../../../../../api/pro_api/SIMT-API/execution/index.md) are used to query the thread hierarchy of the current launch and the position of the current thread:

| API | Description |
|---|---|
| pypto_pro.language.simt.grid_dim() | Number of thread blocks in each dimension of the Grid. |
| pypto_pro.language.simt.block_dim() | Number of threads in each dimension of the thread block. |
| pypto_pro.language.simt.block_idx() | Coordinate of the current thread block in the Grid. |
| pypto_pro.language.simt.thread_idx() | Coordinate of the current thread in the thread block. |
| pypto_pro.language.simt.linear_thread_idx() | Index of the current thread within the block, flattened in X-dimension-first order, excluding the thread block offset. |

The first four APIs return a three-dimensional object in dim3 form, with each component being a DT_UINT32 Scalar; pypto_pro.language.simt.linear_thread_idx() returns a DT_UINT32 Scalar.

**Thread Index in SIMT Programming**

For a one-dimensional Grid and one-dimensional thread block, the global index equals "thread block X coordinate × threads per block + thread X coordinate". For a three-dimensional configuration, the thread block coordinates are first flattened, and then the linear thread index within the block is added:

```python
import pypto_pro.language as pl


@pl.jit(mode="simt", max_threads=64)
def copy_grid(
    source: pl.Tensor[[1536], pl.DT_FP32],
    output: pl.Tensor[[1536], pl.DT_FP32],
):
    grid = pl.simt.grid_dim()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    linear_block = block_id.x + grid.x * (block_id.y + grid.y * block_id.z)
    threads_per_block = block.x * block.y * block.z
    index = linear_block * threads_per_block + pl.simt.linear_thread_idx()
    output[index] = source[index]
```

When launched with `copy_grid[(2, 3, 4), (8, 4, 2)](source, output)`, 24 thread blocks each contain 64 threads, covering 1536 elements in total. In this example, the launch range is the same as the Tensor size; when processing other sizes, you must check whether the index is smaller than the data range before accessing elements.

**Thread Index in Mixed Programming**

In SIMD and SIMT mixed programming, the Grid uses only the X dimension, and the Y and Z coordinates of the thread block are both 0. Each thread block can still be configured with two-dimensional or three-dimensional threads, and pypto_pro.language.simt.linear_thread_idx() is used to flatten the coordinates within the block:

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

When the outer kernel is called through `copy_3d[8, 8, 4](...)`, each thread block contains 256 threads. For example, when 4 Vector cores are actually effective, the 4 thread blocks process indices 0 to 255, 256 to 511, 512 to 767, and 768 to 1023, respectively. When the data volume exceeds the total number of threads in this launch, you can process in strides based on the total number of threads. For details, see [Gather example](#simd-and-simt-mixed-programming-example-gather).

### Synchronization

Select the synchronization method based on where the dependency occurs:

| Dependency Scenario | Usage |
|---|---|
| Within the current thread block, each thread completes one phase before entering the next phase | pypto_pro.language.simt.syncthreads. |
| Constrain the memory access order of the current thread | Select pypto_pro.language.simt.threadfence_block or pypto_pro.language.simt.threadfence based on the scope. |
| In mixed programming, the dependency between data movement and SIMT computation | Pipeline synchronization or Auto Mutex in the outer kernel. |

#### Thread Block Synchronization

pypto_pro.language.simt.syncthreads sets a barrier for the current thread block. For example, in mixed programming, when each thread first writes to a shared UB Tile and then reads data written by other threads, a barrier should be set between the write and read phases. For a complete example, see [pypto_pro.language.simt.syncthreads](../../../../../api/pro_api/SIMT-API/synchronization/syncthreads.md).

All threads in the same thread block must reach the same synchronization point. When there is a tail boundary check, threads without valid data should skip the read/write operations and continue to reach the barrier that requires the participation of the entire block; these threads must not return early or skip the branch containing the barrier. A barrier inside a loop also requires each thread to execute it the same number of times.

This API synchronizes only the current thread block and does not provide a barrier for the entire Grid. A barrier in a SIMT helper also requires the calling threads to satisfy the consistent arrival condition.

#### Memory Fence

A memory fence constrains the memory access order of the calling thread and does not wait for other threads to reach the same location:

- [pypto_pro.language.simt.threadfence_block](../../../../../api/pro_api/SIMT-API/synchronization/threadfence_block.md): Establishes the memory access order within the thread block scope.
- [pypto_pro.language.simt.threadfence](../../../../../api/pro_api/SIMT-API/synchronization/threadfence.md): Establishes the memory access order within the device scope.

For example, when a thread first writes data and then publishes the completion status through an atomic variable, you need to consider the order between the data write and the completion status update, as well as how the consumer observes the completion status. A memory fence itself neither notifies the consumer nor guarantees the atomicity of concurrent updates, and must be used together with a complete thread cooperation protocol. For the mechanism and examples, see [SIMT Synchronization Introduction](../../../../../api/pro_api/SIMT-API/synchronization/synchronization_intro.md).

#### Outer Pipeline Synchronization

In SIMD and SIMT mixed programming, SIMT VF calls are executed asynchronously in the V pipeline. Data moved in by MTE2 must be ready before it can be accessed by SIMT; UB data updated by SIMT must be computed before it can be moved out by MTE3:

```text
load (MTE2) → MTE2/V synchronization → SIMD or SIMT compute (V) → V/MTE3 synchronization → store (MTE3)
```

For ordinary tiles, dependencies are expressed through the paired pypto_pro.language.system.sync_src and pypto_pro.language.system.sync_dst. For the complete code, see [pypto_pro.language.system.sync_src call examples](../../../../../api/pro_api/SIMD-API/synchronization/sync_src.md#examples).

When using pypto_pro.language.make_tile_group with mutex_ids, the Auto Mutex enabled by default can manage the buffer dependencies of pypto_pro.language.load, SIMT entrypoint function calls, and pypto_pro.language.store. Enabling only auto_mutex=True does not automatically complete synchronization for ordinary pypto_pro.language.make_tile.

### Intra-Warp Collaboration

When you need to query thread state, exchange data, or perform reduction within a warp, use the [Warp API](../../../../../api/pro_api/SIMT-API/warp/index.md). The participating threads, source Lane, and data type of the Warp API must satisfy the constraints of each API. Intra-warp collaboration cannot replace synchronization of the entire thread block.

### Mathematical Functions and Type Conversion

SIMT functions can use common Scalar expressions and also provide the following common functions. Only some APIs are listed here:

- Type conversion: for example, pypto_pro.language.simt.cast.
- Basic mathematical operations: for example, pypto_pro.language.simt.abs and pypto_pro.language.simt.sqrt.
- Exponential, logarithmic, and trigonometric functions: for example, pypto_pro.language.simt.exp, pypto_pro.language.simt.log, and pypto_pro.language.simt.sin.
- Rounding operations: for example, pypto_pro.language.simt.floor.
- Floating-point value judgment: for example, pypto_pro.language.simt.isnan.

For the complete APIs and usage constraints, see [Scalar Compute and Type Conversion APIs](../../../../../api/pro_api/SIMT-API/scalar_compute/index.md).

### Atomic Operations

When multiple threads accumulate to the same element, update an extremum, or modify a state, the ordinary "read, compute, write back" pattern may lose updates from other threads. For example, when multiple threads execute counter[0, 0] = counter[0, 0] + 1 simultaneously, the final result is not guaranteed to equal the number of participating threads.

In this case, you can use the [atomic operation API](../../../../../api/pro_api/SIMT-API/atomic/index.md). For example, when counter is a DT_INT32 Tensor, you can call `pypto_pro.language.simt.atomic_add(counter, [0, 0], 1)` inside the SIMT function body to perform an atomic accumulation:

- target receives the container, and offset receives the per-axis element coordinates, for example [0, 0]. The target of a SIMT kernel is a GM Tensor, and mixed programming can also use a UB Tile.
- The data type, input value, and return value of an atomic operation must satisfy the corresponding API constraints. Some types do not return the value before the update.
- Shared states such as counters must be initialized before concurrent updates begin, for example by preparing a zero-value Tensor on the Host side.
- An atomic operation protects the update of the target element. It does not provide a cross-thread-block barrier, nor does it guarantee that data at other addresses is ready.

Atomic operations are commonly used for counting, histograms, and shared state updates. A large number of threads competing for the same address may reduce throughput. You can reduce the number of shared updates based on the algorithm.

## Complete Programming Example

### SIMT Programming Example: Per-Thread Affine Transformation

The following example processes 1000 FP32 elements and computes output = input_tensor × scale + bias. Each thread processes one element, and the SIMT helper reuses the per-thread computation logic:

1. Each thread block is configured with 256 threads, and the Host launches 4 thread blocks.
2. The element index is computed from the thread block number and the thread number within the block.
3. The Tensor is read and written only when the index is smaller than the number of elements. The 24 threads in the last thread block skip the read and write.
4. The Host waits for the Device to finish execution and then verifies the result.

```python
import pypto_pro.language as pl
import torch
import torch_npu


ELEMENTS = 1000
THREADS = 256


@pl.vector_function(mode="simt")
def affine_simt(value: pl.DT_FP32, scale: pl.DT_FP32, bias: pl.DT_FP32) -> pl.DT_FP32:
    return value * scale + bias


@pl.jit(mode="simt", max_threads=THREADS)
def transform_simt_kernel(
    output: pl.Tensor[[ELEMENTS], pl.DT_FP32],
    input_tensor: pl.Tensor[[ELEMENTS], pl.DT_FP32],
    count: pl.DT_UINT32,
    scale: pl.DT_FP32,
    bias: pl.DT_FP32,
):
    index = pl.simt.block_idx().x * pl.simt.block_dim().x + pl.simt.linear_thread_idx()
    if index < count:
        output[index] = affine_simt(input_tensor[index], scale, bias)


torch.npu.set_device(0)
input_tensor = torch.arange(ELEMENTS, dtype=torch.float32, device="npu:0")
output = torch.empty_like(input_tensor)
grid = (ELEMENTS + THREADS - 1) // THREADS
transform_simt_kernel[grid, THREADS](output, input_tensor, ELEMENTS, 2.0, 1.0)
torch.npu.synchronize()
torch.testing.assert_close(output, input_tensor * 2.0 + 1.0, rtol=0, atol=0)
```

In this example, each thread writes to a different element, so there is no data dependency between threads and no thread block barrier is required. Tensor access does not automatically add runtime boundary checks, so the function body retains the index < count condition. The caller must ensure that count does not exceed the Tensor size.

### SIMD and SIMT Mixed Programming Example: Gather

The following example reads data from the input Tensor by the row numbers specified in indices:

```text
output[row, col] = input_tensor[indices[0, row], col]
```

This example expresses Gather semantics through Tensor element access. Each thread processes one or more output rows, and a loop is used within a row for copying. This is an example of index expression; actual performance still needs to be measured in combination with data layout and thread mapping.

```python
import pypto_pro.language as pl
import torch
import torch_npu


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


torch.npu.set_device(0)
input_tensor = torch.randn(INPUT_ROWS, WIDTH, dtype=torch.float32, device="npu:0")
indices = torch.randint(INPUT_ROWS, (1, OUTPUT_ROWS), dtype=torch.int32, device="npu:0")
output = torch.empty((OUTPUT_ROWS, WIDTH), dtype=torch.float32, device="npu:0")

gather_kernel[None, BLOCKS](input_tensor, indices, output, OUTPUT_ROWS)
torch.npu.synchronize()

expected = input_tensor.index_select(0, indices.reshape(-1).to(torch.int64))
torch.testing.assert_close(output, expected, rtol=0, atol=0)
```

The caller must ensure that 0 <= row_count <= OUTPUT_ROWS and that all read indices elements are within [0, INPUT_ROWS). The Tensor shape is fixed, while the index values can vary at runtime.

In this example, the Host side sets block_dim to 48 through BLOCKS, requesting the launch of 48 logical Vector cores. BLOCKS must not exceed the number of Vector cores available in the current execution stream, and can be reduced according to the environment. Each actually effective Vector core launches a thread block containing 256 threads. Threads stride over subsequent rows according to the total number of threads in the actual Grid, so all 12288 rows are still covered after reducing BLOCKS. Each thread writes to a different output row, and there is no data dependency between rows, so no thread block barrier is required.
