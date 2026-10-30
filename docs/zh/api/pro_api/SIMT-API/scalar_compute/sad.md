# pypto_pro.language.simt.sad

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

计算两个整数的绝对差并累加addend，计算公式如下。

$$result = |lhs-rhs|+addend$$

## 函数原型

```python
pypto_pro.language.simt.sad(
    lhs: Scalar,
    rhs: Scalar,
    addend: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，数据类型必须与另一个源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_UINT32，数据类型必须与另一个源操作数一致。Tensor或Tile元素需通过下标访问后传入。 |
| addend | 输入 | 累加值，Scalar类型，仅支持DT_UINT32。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

- 只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。
- lhs与rhs必须同为DT_INT32或同为DT_UINT32，不支持混合输入。

## 返回值说明

返回绝对差与addend之和，数据类型为DT_UINT32；结果超出DT_UINT32范围时按低32位截断。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def sad_example(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    addend: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.sad(lhs[0, tid], rhs[0, tid], addend[0, tid])


@pl.jit()
def sad_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_INT32],
    rhs: pl.Tensor[[1, 64], pl.DT_INT32],
    addend: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        sad_example[64](lhs, rhs, addend, output)
```
