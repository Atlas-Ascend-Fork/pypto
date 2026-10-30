# pypto_pro.language.simt.fma_relu

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

先对三个操作数执行融合乘加，再将负结果钳位为零。乘法与加法之间不进行中间舍入，计算公式如下。

$$result = \max(lhs \times rhs + addend, 0)$$

## 函数原型

```python
pypto_pro.language.simt.fma_relu(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| addend | 输入 | 源操作数，Scalar类型，支持DT_FP16和DT_BF16，数据类型必须与其他源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 三个输入必须同为DT_FP16或同为DT_BF16。

## 返回值说明

返回融合乘加后经过ReLU处理的结果，数据类型与输入一致。结果按最近偶数模式舍入；负结果置为零，NaN返回NaN，正向上溢返回正无穷。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def fma_relu_example(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    addend: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.fma_relu(lhs[0, tid], rhs[0, tid], addend[0, tid])


@pl.jit()
def fma_relu_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_FP16],
    rhs: pl.Tensor[[1, 64], pl.DT_FP16],
    addend: pl.Tensor[[1, 64], pl.DT_FP16],
    output: pl.Tensor[[1, 64], pl.DT_FP16],
):
    with pl.section_vector():
        fma_relu_example[64](lhs, rhs, addend, output)
```
