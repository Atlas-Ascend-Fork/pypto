# pypto_pro.language.simt.brev

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

反转value的全部比特位顺序。输入位宽为$N$时，原第$i$位移动到第$N-1-i$位。

## 函数原型

```python
pypto_pro.language.simt.brev(
    value: Scalar,
) -> Scalar
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| value | 输入 | 源操作数，Scalar类型，支持DT_UINT32和DT_UINT64。Tensor或Tile元素需通过下标访问后传入。 |

## 约束说明

只能在由@pypto_pro.language.vector_function(mode="simt")定义的SIMT入口函数或辅助函数中调用。

## 返回值说明

返回比特顺序反转后的整数，数据类型与输入一致。

## 调用示例

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=64)
def brev_example(
    value: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.brev(value[0, tid])


@pl.jit()
def brev_kernel(
    value: pl.Tensor[[1, 64], pl.DT_UINT32],
    output: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    with pl.section_vector():
        brev_example[64](value, output)
```
