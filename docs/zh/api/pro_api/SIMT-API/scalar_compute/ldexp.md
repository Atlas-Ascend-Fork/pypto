# pypto_pro.language.simt.ldexp

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

用2的整数次幂缩放浮点数，不单独计算$2^{exponent}$，计算公式如下。

$$result = value \times 2^{exponent}$$

与[pypto_pro.language.simt.scalbn](scalbn.md)的计算语义相同；本接口的exponent仅支持DT_INT32，后者还支持DT_INT64。

## 函数原型

```python
pypto_pro.language.simt.ldexp(
    value: Scalar,
    exponent: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 待缩放值，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| exponent | 输入 | 二进制指数，Scalar类型，仅支持DT_INT32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回缩放结果，数据类型为DT_FP32。value为正负零、正负无穷或NaN时返回value；exponent为0时返回value。有限结果上溢时返回对应符号的无穷，下溢至零时返回对应符号的零；可表示的次正规结果仍可返回非零值。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def ldexp_example(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    exponent: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.ldexp(value[0, tid], exponent[0, tid])


@pl.jit()
def ldexp_kernel(
    value: pl.Tensor[[1, 64], pl.DT_FP32],
    exponent: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        ldexp_example[64](value, exponent, output)
```
