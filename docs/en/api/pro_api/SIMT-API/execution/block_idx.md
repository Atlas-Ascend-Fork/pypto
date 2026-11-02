# pypto_pro.language.simt.block_idx

<!-- md-trans-meta sourceCommit=c9f99797c095bdf931d4de1f608ada67b6905876 translatedAt=2026-10-26T15:01:53.716Z pushedAt=2026-10-28T11:57:29.383Z nl=2 -->

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

Obtains the 3D coordinates of the current SIMT thread block in the thread block grid.

Thread block coordinates are used to distinguish different thread blocks in the thread block grid. All threads within the same thread block read the same block_idx(), and different thread blocks process their respective data through different coordinates.

## Prototype

```python
pypto_pro.language.simt.block_idx() -> Any
```

## Parameters

None

## Constraints

This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns a 3D thread block coordinate tuple object. Read the coordinates of each dimension through block.x, block.y, and block.z, where each component is a Scalar of the DT_UINT32 type. The current thread block grid uses only the X dimension. The range of block.x is [0, grid_dim().x), and both block.y and block.z are 0.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_block_idx_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    block = pl.simt.block_idx()
    output[0, tid] = block.x


@pl.jit()
def simt_block_idx_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_block_idx_x[256](output)
```
