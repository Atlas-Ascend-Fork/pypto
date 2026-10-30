# pypto_pro.language.simt.rnorm4d

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

以缩放方式计算四个分量欧几里得范数的倒数。

$$result = \frac{1}{\sqrt{x^2+y^2+z^2+w^2}}$$

## 函数原型

```python
pypto_pro.language.simt.rnorm4d(
    x: Scalar,
    y: Scalar,
    z: Scalar,
    w: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| x | 输入 | 分量，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| y | 输入 | 分量，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| z | 输入 | 分量，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| w | 输入 | 分量，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回四个分量欧几里得范数的倒数，数据类型为DT_FP32。四个输入均为零时返回正无穷；任一输入为无穷时返回正零，即使另有输入为NaN；无无穷输入且任一输入为NaN时返回NaN。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def rnorm4d_example(
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    z: pl.Tensor[[1, 64], pl.DT_FP32],
    w: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.rnorm4d(x[0, tid], y[0, tid], z[0, tid], w[0, tid])


@pl.jit()
def rnorm4d_kernel(
    x: pl.Tensor[[1, 64], pl.DT_FP32],
    y: pl.Tensor[[1, 64], pl.DT_FP32],
    z: pl.Tensor[[1, 64], pl.DT_FP32],
    w: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        rnorm4d_example[64](x, y, z, w, output)
```
