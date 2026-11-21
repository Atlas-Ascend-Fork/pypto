# Introduction

In SIMT programs, the execution progress and memory access completion order of different threads may be different. When there are data dependencies between threads, synchronization barriers or memory fences are required to restrict the execution order and memory visibility.

The synchronization mechanisms are classified into the following two types:

- **Barrier**: A thread can continue to execute only after all threads in the same scope reach the specified positions.
- **Memory fence**: It restricts the memory access order of the calling thread before and after the fence, making the memory operations before the fence visible to other threads within a specified range. A memory fence does not wait for other threads to reach the same position.

## Barrier

pypto_pro.language.simt.syncthreads is used for phase synchronization within a thread block. The following uses the shared UB tile in SIMD and SIMT hybrid programming as an example. Multiple threads first write data, and then enter the next phase to read the data after all threads complete the writing.

```python
shared[0, tid] = value

# Waits for all threads in the current thread block to complete the shared memory write.
pl.simt.syncthreads()

result = shared[0, peer_tid]
```

Each thread writes its own data first. Threads that reach syncthreads stop execution until other threads in the current thread block also reach the same synchronization point. After all threads pass the synchronization point, they can safely read the shared data written by other threads before the synchronization point. For details about the complete calling method, see [pypto_pro.language.simt.syncthreads](syncthreads.md).

## Memory Fence

A memory fence is used to restrict the memory access order of the calling thread. It is usually used to solve the problem of "writing data first and then publishing the flag indicating that the data is ready".

Assume that the producer thread performs the following two write operations in order:

```python
data[0] = value
ready[0] = 1
```

The order in the code does not necessarily mean that other threads observe the two writes in the same order. If the consumer thread has observed ready[0] == 1, but the new value of data[0] is not yet visible in the corresponding scope, the consumer may still read the old data.

Inserting a memory fence before publishing the flag can establish the required memory order:

```python
data[0] = value

# This ensures that the previous data write is visible to other threads before the subsequent update of ready.
pl.simt.threadfence()

pl.simt.atomic_exch(ready[0], 1)
```

The order established by the preceding code is as follows:

```text
          Data write
             ↓
threadfence ensures that the previous write is visible to other threads.
             ↓
Update ready to 1 by using atomic operations.
```

When the consumer observes through the synchronization protocol that ready has been updated, the data written by the producer before the fence should also be visible within the corresponding scope.

Note that the memory fence only restricts the memory access order and visibility of **the calling thread itself**.

- It does not wait for other threads to reach the fence position.
- It does not notify the consumer that the data is ready. The consumer still needs to determine the status by using atomic variables, polling, or other synchronization methods.
- It does not guarantee the atomicity when multiple threads concurrently update the same address. Atomic operations are still required in case of write conflicts.
- It is not equivalent to a thread barrier and cannot replace syncthreads.

The difference between threadfence_block and threadfence lies in their scope of application:

- threadfence_block is used to restrict the memory access order within the current thread block, and is suitable for scenarios where data is shared within the block and UB is used for collaboration.
- threadfence is used to restrict the memory access order within the device, and is suitable for scenarios where data or flags are published to other thread blocks through the GM.

For details about the complete API call example, see [pypto_pro.language.simt.threadfence_block](threadfence_block.md) and [pypto_pro.language.simt.threadfence](threadfence.md).
