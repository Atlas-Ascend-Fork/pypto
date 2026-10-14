# pypto_pro.language.simt.nextafter

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

返回从value沿direction方向的下一个可表示DT_FP32数值；两者相等时返回value。

## 函数原型

```python
pypto_pro.language.simt.nextafter(
    value: float,
    direction: float,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 起始值，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| direction | 输入 | 目标方向，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT函数中调用。

## 返回值说明

返回相邻的可表示浮点值，数据类型为DT_FP32。特殊值如下：

| value、direction取值 | 返回值 |
|---|---|
| 任一输入为NaN | NaN |
| 两者数值相等（包括符号不同的零） | value |
| value为±0，direction为正数 | 最小正非规格化数 |
| value为±0，direction为负数 | 最小负非规格化数 |
| value为+Inf，direction为有限数 | 最大有限正数 |
| value为-Inf，direction为有限数 | 最小有限负数 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def nextafter_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    direction: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.nextafter(value[0, tid], direction[0, tid])


@pl.jit()
def nextafter_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    direction: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        nextafter_example[64](value, direction, output)
```
