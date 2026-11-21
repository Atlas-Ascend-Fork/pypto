# pypto_pro.language.simt.threadfence_block

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

Sets a memory fence in the thread block scope so that the memory access initiated by the current thread before the fence is visible to threads in the current thread block and precedes the memory access after the fence.

This API only restricts the memory access order of the current thread and does not wait for other threads. It is not a thread barrier.

## Prototype

```python
pypto_pro.language.simt.threadfence_block() -> None
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- It must be called as an independent statement, and the results cannot be used in expressions.
- This API does not guarantee that other threads have reached the same position. To wait for all threads in the current thread block, use [pypto_pro.language.simt.syncthreads](syncthreads.md).

## Return Value

None

## Examples

Each thread writes data to the shared UB, calls threadfence_block to publish the write, and then determines the last thread that completes the write by using the atomic counter.

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def publish_with_threadfence_block(out, values, completed):
    tid = pl.simt.linear_thread_idx()
    if tid == 0:
        completed[0, 0] = 0
    values[0, tid] = 0
    pl.simt.syncthreads()

    values[0, tid] = 1
    pl.simt.threadfence_block()
    ticket = pl.simt.atomic_add(completed[0, 0], 1)
    if ticket == THREADS - 1:
        total: pl.DT_INT32 = 0
        for index in pl.range(THREADS):
            total = total + values[0, index]
        out[0, 0] = total


@pl.jit()
def simt_threadfence_block(
    out: pl.Tensor[[1, 1], pl.DT_INT32],
):
    values = pl.make_tile(
        pl.TileType(
            shape=[1, THREADS],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0,
    )
    completed = pl.make_tile(
        pl.TileType(
            shape=[1, 8],
            dtype=pl.DT_INT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=THREADS * 4,
    )
    with pl.section_vector():
        publish_with_threadfence_block[THREADS](out, values, completed)
```
