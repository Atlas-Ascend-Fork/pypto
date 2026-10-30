# pypto_pro.language.simt.ffs

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

从最低位开始查找value中第一个为1的比特位，并返回从1开始的位置编号。

## 函数原型

```python
pypto_pro.language.simt.ffs(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_INT32和DT_INT64。按其二进制补码位模式进行查找。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回最低有效1位的位置，数据类型为DT_INT32，取值范围为0到输入位宽；value为0时返回0。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def ffs_example(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.ffs(value[0, tid])


@pl.jit()
def ffs_kernel(
    value: pl.Tensor[[1, 64], pl.DT_INT32],
    output: pl.Tensor[[1, 64], pl.DT_INT32],
):
    with pl.section_vector():
        ffs_example[64](value, output)
```
