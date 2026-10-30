# pypto_pro.language.simt.asinh

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

计算value的反双曲正弦。对于非负输入，计算公式如下，负输入按奇函数关系处理。

$$result = \ln\left(value+\sqrt{value^2+1}\right)$$

## 函数原型

```python
pypto_pro.language.simt.asinh(
    value: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT函数中调用。

## 返回值说明

返回反双曲正弦值，数据类型为DT_FP32。特殊值如下：

| value取值 | 返回值 |
|---|---|
| +0 | +0 |
| -0 | -0 |
| +Inf | +Inf |
| -Inf | -Inf |
| NaN | NaN |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def asinh_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.asinh(value[0, tid])


@pl.jit()
def asinh_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        asinh_example[64](value, output)
```
