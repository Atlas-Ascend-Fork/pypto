# pypto_pro.language.simt.block_dim

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Obtains the number of threads in the current SIMT thread block along the X, Y, and Z dimensions.

A thread block is the launch unit of a SIMT function. Threads in a block execute the same SIMT function and process different data using thread indices.

## Prototype

```python
pypto_pro.language.simt.block_dim() -> Any
```

## Parameters

None

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a tuple object of the three-dimensional thread block size. You can read the size of each dimension through dimensions.x, dimensions.y, and dimensions.z. Each component is a scalar of the DT_UINT32 type.

- SIMT programming: The thread block size is determined by the block passed when the host calls `kernel[grid, block](...)` or `kernel[stream, grid, block](...)`. For example, if block is (8, 4, 2), the returned value is (8, 4, 2). For the unspecified dimensions, the value 1 is added.
- SIMD and SIMT hybrid programming: The thread block size is set by the threads in the square brackets when the outer JIT kernel calls `simt_func[threads](...)`. For the unspecified dimensions, the value 1 is added.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_block_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    dimensions = pl.simt.block_dim()
    output[0, tid] = dimensions.x


@pl.jit()
def simt_block_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_block_dim_x[256](output)
```
