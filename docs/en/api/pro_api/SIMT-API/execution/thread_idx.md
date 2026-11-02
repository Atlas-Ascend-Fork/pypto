# pypto_pro.language.simt.thread_idx

<!-- md-trans-meta sourceCommit=c9f99797c095bdf931d4de1f608ada67b6905876 translatedAt=2026-10-26T15:03:04.766Z pushedAt=2026-10-28T11:57:29.396Z nl=2 -->

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

Obtains the 3D coordinates of the current thread within the SIMT thread block.

A thread is the smallest programming unit of SIMT execution. thread_idx() returns the local coordinates of the thread within the thread block. Each thread block is numbered from zero, so threads in different thread blocks can have the same thread coordinates.

## Prototype

```python
pypto_pro.language.simt.thread_idx() -> Any
```

## Parameters

None

## Constraints

This API can be called only in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns a 3D thread coordinate tuple object thread. Read the coordinates of each dimension through thread.x, thread.y, and thread.z. Each component is a Scalar of the DT_UINT32 type. The value ranges of the three components are [0, block_dim().x), [0, block_dim().y), and [0, block_dim().z), respectively.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_thread_idx_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    thread = pl.simt.thread_idx()
    output[0, tid] = thread.x


@pl.jit()
def simt_thread_idx_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_thread_idx_x[256](output)
```
