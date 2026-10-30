# pypto_pro.language.simt.threadfence

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

设置设备作用域的内存栅栏，使当前线程在栅栏前发起的内存访问先于栅栏后的内存访问，对设备上的线程可见。

该接口只约束当前线程的内存访问顺序，不等待当前线程块或其他线程块中的线程。

## 函数原型

```python
pypto_pro.language.simt.threadfence() -> None
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- 必须作为独立语句调用，不支持使用其结果参与表达式。
- 该接口不是设备级线程屏障。如需在线程块内等待所有线程，应使用[pypto_pro.language.simt.syncthreads](syncthreads.md)。

## 返回值说明

无。

## 调用示例

每个线程块先写入自己的GM数据，再调用threadfence发布写入，随后通过GM中的原子计数更新完成状态。最后完成的线程块可以读取其他线程块已经写入的数据。

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

# 使用4个线程块启动。
# simt_threadfence[None, BLOCKS](out, values, completed)
```
