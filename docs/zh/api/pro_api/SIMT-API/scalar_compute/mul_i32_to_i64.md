# pypto_pro.language.simt.mul_i32_to_i64

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

计算两个同类型32位整数的完整64位乘积。

$$result = lhs \times rhs$$

## 函数原型

```python
pypto_pro.language.simt.mul_i32_to_i64(
    lhs: Scalar,
    rhs: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，两个输入的数据类型必须一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，两个输入的数据类型必须一致。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- 不支持DT_INT32与DT_UINT32混合输入。

## 返回值说明

返回完整乘积，输入和输出类型的对应关系如下。

| lhs、rhs数据类型 | 返回数据类型 |
|---|---|
| DT_INT32 | DT_INT64 |
| DT_UINT32 | DT_UINT64 |

结果保留全部64位；与[pypto_pro.language.simt.mul_hi](mul_hi.md)只返回高半部分不同，也不按[pypto_pro.language.simt.mul24](mul24.md)截取低24位操作数。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def mul_i32_to_i64_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT64],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.mul_i32_to_i64(lhs[0, tid], rhs[0, tid])


@pl.jit()
def mul_i32_to_i64_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT64],
):
    with pl.section_vector():
        mul_i32_to_i64_example[64](lhs, rhs, output)
```
