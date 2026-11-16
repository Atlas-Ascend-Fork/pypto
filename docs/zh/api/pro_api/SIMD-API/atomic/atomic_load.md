# pypto_pro.language.atomic_load

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

原子地读取Global Memory中`tensor`指定位置的元素并返回该值。该读取绕过Scalar DCache，可观察到其他核通过原子写或DCache旁路写发布的最新值。

## 函数原型

```python
pypto_pro.language.atomic_load(
    tensor: Tensor,
    offset: int,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| tensor | 输入 | 源操作数，GM Tensor（kernel参数或`pypto_pro.language.tensor_view`/`pypto_pro.language.make_tensor`视图），通过`offset`指定要读取的元素。支持DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| offset | 输入 | 元素位置，以元素为单位的线性偏移，与[getval](../transpose_and_element_access/getval.md)一致。支持整型常量或运行时整型标量表达式（包括循环变量）；取值范围0 ≤ offset < 总元素数，越界行为不确定。 |

## 约束说明

公共约束（调用位置、tensor限制、DCache一致性与顺序）见[标量原子操作概述](atomic_overview.md)，本接口无额外约束。

## 返回值说明

返回读取到的元素值，类型与元素类型一致。可作为独立语句调用以丢弃返回值。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def atomic_load_kernel(state: pl.Tensor[[64], pl.DT_INT32]):
    with pl.section_vector():
        value = pl.atomic_load(state, 0)
        pl.setval(state, 1, value)
```
