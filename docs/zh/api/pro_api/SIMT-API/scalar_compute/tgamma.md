# pypto_pro.language.simt.tgamma

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

计算Gamma函数。对于正实数，定义如下，并通过延拓覆盖非整数负数。

$$result = \Gamma(value)=\int_0^{+\infty}t^{value-1}e^{-t}\,dt$$

## 函数原型

```python
pypto_pro.language.simt.tgamma(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- Gamma函数在零和负整数处存在极点。

## 返回值说明

返回Gamma函数值，数据类型为DT_FP32。正负零分别返回正负无穷，负整数和负无穷返回NaN，正无穷返回正无穷，NaN返回NaN。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def tgamma_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.tgamma(value[0, tid])


@pl.jit()
def tgamma_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        tgamma_example[64](value, output)
```
