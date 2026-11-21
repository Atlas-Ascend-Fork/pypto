# pypto_pro.language.simt.syncthreads

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

Sets a barrier at the thread block level. The threads continue to execute only after all threads in the current thread block reach the barrier. Writes to shared memory before the barrier are visible to threads in the current thread block after the barrier.

## Prototype

```python
pypto_pro.language.simt.syncthreads() -> None
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- It must be called as an independent statement, and the results cannot be used in expressions.
- Ensure that all threads in the current thread block can reach the same syncthreads synchronization point. When there are multiple synchronization points, all threads must execute these synchronization points.
- When this API is used in a control flow such as if, for, or while, ensure that all threads can reach the synchronization point. For example, a conditional branch containing the synchronization point must be entered by all threads in a consistent manner. The synchronization point in a loop must be executed by all threads for the same number of times. Otherwise, some threads may skip the synchronization point due to break, continue, or an early return.
- This API only synchronizes threads in the current thread block and does not synchronize threads in different thread blocks.

## Return Value

None

## Examples

Each thread writes its own thread ID to the shared UB. After syncthreads is called, the data written by another thread is read.

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def exchange_after_syncthreads(
    out: pl.Tensor[[1, THREADS], pl.DT_UINT32],
    shared,
):
    tid = pl.simt.linear_thread_idx()
    shared[0, tid] = tid
    pl.simt.syncthreads()
    out[0, tid] = shared[0, THREADS - 1 - tid]


@pl.jit()
def simt_syncthreads(out: pl.Tensor[[1, THREADS], pl.DT_UINT32]):
    shared = pl.make_tile(
        pl.TileType(
            shape=[1, THREADS],
            dtype=pl.DT_UINT32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0,
    )
    with pl.section_vector():
        exchange_after_syncthreads[THREADS](out, shared)
```
