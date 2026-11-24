# pypto_pro.language.scalar_abs

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

在Kernel代码中可直接使用Python内置abs(x)，编译时会自动转换为pypto_pro.language.scalar_abs(x)，取单个标量的绝对值|x|。

## 函数原型

```python
pypto_pro.language.scalar_abs(
    x: Union[int, float],
    /,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| x | 输入 | 源操作数（Python int、Python float或Kernel内整型或浮点型标量表达式），支持DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_FP16、DT_BF16、DT_FP32；DT_BOOL操作数按DT_INT64参与计算。 |

## 返回值说明

返回x的绝对值|x|，为标量值表达式（类型与操作数一致）；x为DT_BOOL时返回DT_INT64标量。

## 约束说明

- **适用范围**：本接口仅用于标量取绝对值，常用于循环边界、索引计算；针对Tile逐元素取绝对值请使用[pypto_pro.language.abs](../../SIMD-API/tile_computation/elementwise/abs.md)。

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    count = abs(start - end)
    bound = pl.scalar_abs(total - i)

    for _ in pl.range(bound):
        ...
```
