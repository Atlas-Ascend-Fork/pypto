# pypto_pro.language.simt.linear_thread_idx

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:02:50.591Z pushedAt=2026-10-28T11:57:29.393Z nl=2 -->

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

Obtains the one-dimensional index of the current thread within the SIMT thread block, expanded with X-dimension priority.

The three-dimensional thread coordinates are expanded in X-dimension-priority order. The computation formula is as follows:

$$result = thread\_idx().x + thread\_idx().y \times block\_dim().x + thread\_idx().z \times block\_dim().x \times block\_dim().y$$

## Prototype

```python
pypto_pro.language.simt.linear_thread_idx() -> Scalar
```

## Parameters

None

## Constraints

This API can only be called within a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

## Return Value

Returns a Scalar of the DT_UINT32 type, representing the one-dimensional index of the current thread within the SIMT thread block. The value ranges from 0 to the total number of threads in the current thread block minus 1.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def copy_by_linear_thread_idx(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = source[0, tid]


@pl.jit()
def simt_linear_thread_idx_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_FP32],
):
    with pl.section_vector():
        copy_by_linear_thread_idx[256](source, output)
```
