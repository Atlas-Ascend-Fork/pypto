# pypto_pro.language.simt.max_nan

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

比较两个低精度浮点数并返回较大值；任一输入为NaN时传播该NaN。

## 函数原型

```python
pypto_pro.language.simt.max_nan(
    lhs: float,
    rhs: float,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

1. 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT函数中调用。
2. 两个输入必须同为DT_FP16或同为DT_BF16。

## 返回值说明

返回较大值，数据类型与输入一致。任一输入为NaN时返回NaN；两个输入分别为正零和负零时返回正零。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def max_nan_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.max_nan(lhs[0, tid], rhs[0, tid])


@pl.jit()
def max_nan_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        max_nan_example[64](lhs, rhs, output)
```
