# pypto_pro.language.simt.min_nan

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

比较两个低精度浮点数并返回较小值；任一输入为NaN时传播该NaN。

## 函数原型

```python
pypto_pro.language.simt.min_nan(
    lhs: float,
    rhs: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，浮点型常量或运行时浮点标量表达式，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在SIMT kernel、SIMT vector function或SIMT helper中调用。
- 两个输入必须同为DT_FP16或同为DT_BF16。

## 返回值说明

返回较小值，数据类型与输入一致。lhs为NaN时返回lhs；否则rhs为NaN时返回rhs，两个输入均为NaN时优先返回lhs。

DT_FP16输入均不为NaN且比较相等时返回lhs，混合正负零时保留lhs的符号。DT_BF16输入分别为正零和负零时返回负零，与输入顺序无关。

| lhs | rhs | DT_FP16返回值 | DT_BF16返回值 |
|---|---|---|---|
| 正零 | 负零 | 正零 | 负零 |
| 负零 | 正零 | 负零 | 负零 |

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def min_nan_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.min_nan(lhs[0, tid], rhs[0, tid])


@pl.jit()
def min_nan_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        min_nan_example[64](lhs, rhs, output)
```
