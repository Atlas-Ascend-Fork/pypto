# pypto_pro.language.simt.warp_size

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

获取目标硬件一个SIMT Warp包含的线程数。

## 函数原型

```python
pypto_pro.language.simt.warp_size() -> Scalar
```

## 参数说明

无。

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回目标硬件的Warp线程数，数据类型为DT_INT32。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=1)
def write_warp_size(output: pl.Tensor[[1, 1], pl.DT_INT32]):
    output[0, 0] = pl.simt.warp_size()


@pl.jit()
def warp_size_kernel(output: pl.Tensor[[1, 1], pl.DT_INT32]):
    with pl.section_vector():
        write_warp_size[1](output)
```
