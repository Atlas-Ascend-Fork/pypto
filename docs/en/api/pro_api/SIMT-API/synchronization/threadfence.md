# pypto_pro.language.simt.threadfence

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

Sets a memory fence in the device scope so that the memory access initiated by the current thread before the fence is visible to threads on the device and precedes the memory access after the fence.

This API only restricts the memory access order of the current thread and does not wait for threads in the current thread block or other thread blocks.

## Prototype

```python
pypto_pro.language.simt.threadfence() -> None
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- It must be called as an independent statement, and the results cannot be used in expressions.
- This API is not a device-level thread barrier. To wait for all threads in a thread block, use [pypto_pro.language.simt.syncthreads](syncthreads.md).

## Return Value

None

## Examples

Each thread block writes its own GM data, calls threadfence to publish the write, and then updates the completion status through the atomic count in the GM. The thread block that completes the operation last can read the data written by other thread blocks.

```python
import pypto_pro.language as pl

BLOCKS = 4


@pl.vector_function(mode="simt", max_threads=1)
def publish_with_threadfence(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    values: pl.Tensor[[1, BLOCKS], pl.DT_INT32],
    completed: pl.Tensor[[1, 1], pl.DT_INT32],
):
    block_id = pl.simt.block_idx().x
    values[0, block_id] = 1
    pl.simt.threadfence()
    ticket = pl.simt.atomic_add(completed[0, 0], 1)
    if ticket == BLOCKS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(BLOCKS):
            total = total + values[0, index]
        out[0, 0] = total


@pl.jit()
def simt_threadfence(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
    values: pl.Tensor[[1, BLOCKS], pl.DT_INT32],
    completed: pl.Tensor[[1, 1], pl.DT_INT32],
):
    with pl.section_vector():
        publish_with_threadfence[1](out, values, completed)

# Use four thread blocks for startup.
# simt_threadfence[None, BLOCKS](out, values, completed)
```
