# pypto_pro.language.simt.pow

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

计算以lhs为底、rhs为指数的幂。

$$result = lhs^{rhs}$$

## 函数原型

```python
pypto_pro.language.simt.pow(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 底数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 指数，Scalar类型，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 两个输入均须为DT_FP32；不支持整数或低精度浮点类型混合输入。
- 有限负数底数与有限非整数指数的组合返回NaN，不进行输入数值范围的编译期校验。

## 返回值说明

返回幂运算结果，数据类型为DT_FP32。负数底数的有限整数次幂在指数为奇数时为负，在指数为偶数时为正。结果溢出时返回相应符号的无穷，下溢至零时保留相应符号。

以下规则按表中顺序匹配。

| 输入情况 | 返回值 |
|---|---|
| rhs为±0，或lhs为1（另一输入可为NaN） | 1 |
| 其余包含NaN的输入 | NaN |
| lhs为有限负数，rhs为有限非整数 | NaN |
| lhs为±0，rhs > 0 | rhs为奇整数时返回与lhs同符号的零，否则返回+0 |
| lhs为±0，rhs < 0 | rhs为奇整数时返回与lhs同符号的无穷，否则返回+Inf |
| lhs为±Inf，rhs > 0 | rhs为奇整数时返回与lhs同符号的无穷，否则返回+Inf |
| lhs为±Inf，rhs < 0 | rhs为奇整数时返回与lhs同符号的零，否则返回+0 |
| lhs为-1，rhs为±Inf | 1 |
| $\vert lhs\vert >1$，rhs为+Inf；或$\vert lhs\vert <1$，rhs为-Inf | +Inf |
| $\vert lhs\vert <1$，rhs为+Inf；或$\vert lhs\vert >1$，rhs为-Inf | +0 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def pow_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.pow(lhs[0, tid], rhs[0, tid])


@pl.jit()
def pow_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP32],
    rhs: pl.Tensor[[1, 64], pl.DT_FP32],
    output: pl.Tensor[[1, 64], pl.DT_FP32],
):
    with pl.section_vector():
        pow_example[64](lhs, rhs, output)
```
