# pypto_pro.language.simt.threadfence_block

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

设置线程块作用域的内存栅栏，使当前线程在栅栏前发起的内存访问先于栅栏后的内存访问，对当前线程块中的线程可见。

该接口只约束当前线程的内存访问顺序，不等待其他线程，不是线程屏障。

## 函数原型

```python
pypto_pro.language.simt.threadfence_block() -> None
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- 必须作为独立语句调用，不支持使用其结果参与表达式。
- 该接口不保证其他线程已经执行到同一位置。如需等待当前线程块中的所有线程，应使用[pypto_pro.language.simt.syncthreads](syncthreads.md)。

## 返回值说明

无。

## 调用示例

每个线程先写共享UB，再调用threadfence_block发布写入，最后通过原子计数确定最后完成写入的线程。

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
