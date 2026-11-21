# pypto_pro.language.simt.lane_id

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

Obtains the lane ID of the current thread in the warp that the thread belongs to. The number of threads in a warp is fixed at 32. Therefore, the lane ID ranges from 0 to 31.

## Prototype

```python
pypto_pro.language.simt.lane_id() -> Scalar
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the lane ID of the current thread. The data type is DT_INT32, and the value range is [0, 31].

## Examples

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def lane_id_example(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.lane_id()


@pl.jit()
def lane_id_kernel(output: pl.Tensor[[1, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        lane_id_example[WARP_SIZE](output)
```
