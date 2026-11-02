# pypto_pro.language.simt.grid_dim

<!-- md-trans-meta sourceCommit=c9f99797c095bdf931d4de1f608ada67b6905876 translatedAt=2026-10-26T15:02:06.991Z pushedAt=2026-10-28T11:57:29.390Z nl=2 -->

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

Obtains the number of thread blocks in the X, Y, and Z dimensions of the current thread block grid.

The size of the thread block grid comes from the Vector execution domain that actually takes effect in the outer kernel, and may be affected by the runtime time-limit core. The total number of thread blocks in the grid is the product of the sizes in the X, Y, and Z dimensions.

## Prototype

```python
pypto_pro.language.simt.grid_dim() -> Any
```

## Parameters

None

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns a tuple object representing the three-dimensional thread block grid size. Read the size of each dimension through grid.x, grid.y, and grid.z, where each component is a Scalar of the DT_UINT32 type. The current thread block grid uses only the X dimension, where grid.x is the number of logical Blocks in the Vector execution domain of the outer kernel, and both grid.y and grid.z are 1.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_grid_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    grid = pl.simt.grid_dim()
    output[0, tid] = grid.x


@pl.jit()
def simt_grid_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_grid_dim_x[256](output)
```
