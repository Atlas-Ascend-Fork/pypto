# pypto_pro.language.simt.block_dim

<!-- md-trans-meta sourceCommit=c9f99797c095bdf931d4de1f608ada67b6905876 translatedAt=2026-10-26T15:01:17.068Z pushedAt=2026-10-28T11:57:29.380Z nl=2 -->

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

A thread block is the launch unit of a SIMT function. Threads within a block execute the same SIMT function and process different data through their thread indices.

## Prototype

```python
pypto_pro.language.simt.block_dim() -> Any
```

## Parameters

None

## Constraints

This API can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns a three-dimensional thread block size tuple object. The thread block size is set by the outer JIT Kernel through the `threads` in square brackets when calling `simt_func[threads](...)`. Read the size of each dimension through dimensions.x, dimensions.y, and dimensions.z. Each component is a Scalar of the DT_UINT32 type.

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
