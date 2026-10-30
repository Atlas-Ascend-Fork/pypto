# pypto_pro.language.simt.fmod

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

计算两个DT_FP32标量相除的浮点余数。对于有限输入且rhs不为0，其数学语义如下：

$$result = lhs - \operatorname{trunc}(lhs / rhs) \times rhs$$

公式中的trunc表示将中间商lhs / rhs向零取整。结果的符号与被除数lhs一致，绝对值小于除数rhs的绝对值。

该接口与Python的%运算符语义不同。例如，fmod(-5.5, 2.0)为-1.5，而Python表达式-5.5 % 2.0的结果为0.5。

## 函数原型

```python
pypto_pro.language.simt.fmod(
    lhs: float,
    rhs: float,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 被除数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 除数，浮点型常量或运行时浮点标量表达式，仅支持DT_FP32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在SIMT kernel、SIMT vector function或SIMT helper中调用。

## 返回值说明

返回DT_FP32类型的Scalar。特殊输入的行为如下：

| 输入情况 | 返回结果 |
|---|---|
| 任意操作数为NaN | NaN |
| lhs为正负无穷，或rhs为正负零 | NaN |
| lhs为有限数，rhs为正负无穷 | lhs，保留其符号 |
| lhs为正负零，rhs非零且非NaN | 与lhs同符号的零 |
| 有限输入能够整除，且rhs非零 | 与lhs同符号的零 |

## 调用示例

```python
import pypto_pro.language as pl

THREADS = 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def calculate_remainder(
    lhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fmod(lhs[0, tid], rhs[0, tid])


@pl.jit()
def simt_fmod_kernel(
    lhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    rhs: pl.Tensor[[1, THREADS], pl.DT_FP32],
    output: pl.Tensor[[1, THREADS], pl.DT_FP32],
):
    with pl.section_vector():
        calculate_remainder[THREADS](lhs, rhs, output)
```
