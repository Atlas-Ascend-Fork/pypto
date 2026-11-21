# pypto_pro.language.simt.block_idx

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

Obtains the three-dimensional coordinates of the current SIMT thread block in the thread block grid.

Thread block coordinates are used to distinguish different thread blocks in the thread block grid. All threads in the same thread block read the same block_idx(), and different thread blocks process their own data using different coordinates.

## Prototype

```python
pypto_pro.language.simt.block_idx() -> Any
```

## Parameters

None

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a tuple object of the three-dimensional thread block coordinates. You can read the coordinates of each dimension through block.x, block.y, and block.z. Each component is a scalar of the DT_UINT32 type.

- SIMT programming: The coordinates along each dimension are determined by the grid passed by the host. For example, if grid is (2, 3, 4), the coordinate ranges of block.x, block.y, and block.z are [0, 1], [0, 2], and [0, 3], respectively.
- In SIMD and SIMT hybrid programming, the value range of block.x is [0, pypto_pro.language.simt.grid_dim().x), and the values of block.y and block.z are both 0.

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
