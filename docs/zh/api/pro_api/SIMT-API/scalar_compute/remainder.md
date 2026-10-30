# pypto_pro.language.simt.remainder

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

计算lhs除以rhs的IEEE余数。设$n$为最接近$lhs/rhs$的整数，距离相等时选择偶数，计算公式如下。

$$result = lhs-n\times rhs$$

## 函数原型

```python
pypto_pro.language.simt.remainder(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 被除数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 除数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回余数，数据类型为DT_FP32。对于有限输入和非零除数，余数绝对值不超过$|rhs|/2$；余数可以与lhs异号，结果为零时保留lhs的符号。除数为零、被除数为无穷或任一输入为NaN时返回NaN；有限被除数除以无穷时返回被除数。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def remainder_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.remainder(lhs[0, tid], rhs[0, tid])


@pl.jit()
def remainder_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        remainder_example[64](lhs, rhs, output)
```
