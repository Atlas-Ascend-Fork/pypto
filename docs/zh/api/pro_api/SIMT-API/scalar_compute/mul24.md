# pypto_pro.language.simt.mul24

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

取两个32位输入的低24位相乘，并返回乘积的低32位。对于DT_INT32，低24位先按24位有符号整数进行符号扩展；对于DT_UINT32，低24位按无符号整数解释。

## 函数原型

```python
pypto_pro.language.simt.mul24(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，数据类型必须与另一个源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，数据类型必须与另一个源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 不支持DT_INT32与DT_UINT32混合输入。

## 返回值说明

返回低24位操作数乘积的低32位，数据类型与输入一致。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def mul24_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.mul24(lhs[0, tid], rhs[0, tid])


@pl.jit()
def mul24_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        mul24_example[64](lhs, rhs, output)
```
