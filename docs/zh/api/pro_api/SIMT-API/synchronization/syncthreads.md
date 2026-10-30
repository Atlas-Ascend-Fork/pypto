# pypto_pro.language.simt.syncthreads

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

设置线程块级屏障。当前线程块中的所有线程均到达该屏障后，线程才继续执行。屏障前对共享存储的写入在屏障后对当前线程块中的线程可见。

## 函数原型

```python
pypto_pro.language.simt.syncthreads() -> None
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- 必须作为独立语句调用，不支持使用其结果参与表达式。
- 须保证当前线程块中的所有线程都能到达同一个syncthreads同步点，存在多个同步点时，所有线程都要执行这些同步点。
- 在if、for或while等控制流中使用该接口时，须保证所有线程都能到达同步。例如，包含同步点的条件分支须由所有线程一致进入；循环中的同步点须由所有线程执行相同次数，避免部分线程因break、continue或提前return跳过同步点。
- 该接口仅同步当前线程块中的线程，不同步不同线程块。

## 返回值说明

无。

## 调用示例

每个线程先将自己的线程编号写入共享UB。调用syncthreads后，再读取另一个线程写入的数据。

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
