# pypto_pro.language.simt.linear_thread_idx

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

Obtains the one-dimensional index of the current thread in the SIMT thread block. It is flattened in X-major order.

The three-dimensional thread coordinates are flattened in X-major order. The computation formula is as follows:

$$result = thread\_idx().x + thread\_idx().y \times block\_dim().x + thread\_idx().z \times block\_dim().x \times block\_dim().y$$

## Prototype

```python
pypto_pro.language.simt.linear_thread_idx() -> Scalar
```

## Parameters

None

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a scalar of the DT_UINT32 type, representing the one-dimensional index of the current thread in the SIMT thread block. The value starts from 0 and is equal to the total number of threads in the current thread block minus 1.

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
