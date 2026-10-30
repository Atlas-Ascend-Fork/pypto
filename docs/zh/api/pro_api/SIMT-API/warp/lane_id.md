# pypto_pro.language.simt.lane_id

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

获取当前线程在其所属Warp内的Lane ID，当前一个Warp中的线程数量为固定值32，故Lane ID的取值范围为[0,31]

## 函数原型

```python
pypto_pro.language.simt.lane_id() -> Scalar
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回当前线程的Lane ID，数据类型为DT_INT32，取值范围为[0, 31]。

## 调用示例

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
