# pypto_pro.language.simt.copysign

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

将magnitude的绝对值与sign的符号位组合后返回。

## 函数原型

```python
pypto_pro.language.simt.copysign(
    magnitude: float,
    sign: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| magnitude | 输入 | 提供绝对值的源操作数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| sign | 输入 | 符号选择操作数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回绝对值来自magnitude、符号位来自sign的Scalar，数据类型为DT_FP32。sign为零、无穷或NaN时，同样只使用其符号位。特殊值如下：

| magnitude取值 | 返回值 |
|---|---|
| ±0 | 符号位取自sign的零 |
| ±Inf | 符号位取自sign的无穷 |
| NaN | NaN，符号位取自sign |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def copysign_example(
    magnitude: pl.Tensor[[1, 64], pl.DT_FP32],
    sign: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.copysign(magnitude[0, tid], sign[0, tid])


@pl.jit()
def copysign_kernel(
    magnitude: pl.Tensor[[1, 64], pl.DT_FP32],
    sign: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        copysign_example[64](magnitude, sign, output)
```
