# pypto_pro.language.simt.byte_perm

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

将lhs作为低4字节、rhs作为高4字节组成8字节源数据，再由selector依次选择4个字节组成结果。结果的第$i$个字节使用selector第$i$个半字节的低3位作为索引，索引0～3对应lhs的4个字节，索引4～7对应rhs的4个字节。

## 函数原型

```python
pypto_pro.language.simt.byte_perm(
    lhs: Scalar,
    rhs: Scalar,
    selector: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 低4个源字节，Scalar类型，仅支持DT_UINT32。Tensor或Tile元素需通过下标访问后传入。 |
| rhs | 输入 | 高4个源字节，Scalar类型，仅支持DT_UINT32。Tensor或Tile元素需通过下标访问后传入。 |
| selector | 输入 | 字节选择器，Scalar类型，仅支持DT_UINT32。低16位按4个半字节使用，每个半字节的低3位为有效索引。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回按selector排列得到的4字节整数，数据类型为DT_UINT32。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def byte_perm_example(
    lhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    rhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    selector: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.byte_perm(lhs[0, tid], rhs[0, tid], selector[0, tid])


@pl.jit()
def byte_perm_kernel(
    lhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    rhs: pl.Tensor[[1, 64], pl.DT_UINT32],
    selector: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        byte_perm_example[64](lhs, rhs, selector, output)
```
