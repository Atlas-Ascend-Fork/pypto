# pypto_pro.language.atomic_store

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

原子地将`value`写入Global Memory中`tensor`指定位置的元素。该写入绕过Scalar DCache，其他核可通过原子读或DCache旁路读观察到写入后的值。

## 函数原型

```python
pypto_pro.language.atomic_store(
    tensor: Tensor,
    offset: int,
    value: Union[int, float],
) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| tensor | 输出 | 目的操作数，GM Tensor（kernel参数或`pypto_pro.language.tensor_view`/`pypto_pro.language.make_tensor`视图），通过`offset`指定要写入的元素。支持DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64。 |
| offset | 输入 | 元素位置，以元素为单位的线性偏移，与[getval](../transpose_and_element_access/getval.md)一致。支持整型常量或运行时整型标量表达式（包括循环变量）；取值范围0 ≤ offset < 总元素数，越界行为不确定。 |
| value | 输入 | 写入值，整型或浮点型常量，或相应类型的运行时标量表达式。数据类型须与`tensor`元素类型一致；数值字面量按元素类型处理，整数元素不接受浮点字面量。 |

## 约束说明

公共约束（调用位置、tensor限制、DCache一致性与顺序）见[标量原子操作概述](atomic_overview.md)。本接口底层为posted write，不发射`dsb`排空写缓冲；本核后续访存或跨核可见性需由调用方显式插入`dsb`或同步接口建立先后关系。

## 返回值说明

无返回值。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def atomic_store_kernel(state: pl.Tensor[[64], pl.DT_INT32]):
    with pl.section_vector():
        pl.atomic_store(state, 0, 7)
        value = pl.atomic_load(state, 0)
        pl.setval(state, 1, value)
```
