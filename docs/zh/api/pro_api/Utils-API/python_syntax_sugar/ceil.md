# pypto_pro.language.ceil

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

计算单个标量的向上取整，即`res = ceil(x)`，结果为不小于x的最小整数。整数操作数保持不变，浮点操作数（DT_FP16/DT_BF16/DT_FP32）向上取整。

## 函数原型

```python
pypto_pro.language.ceil(
    x: Union[int, float],
    /,
) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| x | 输入 | 源操作数（Python int、Python float或Kernel内整型或浮点型标量表达式），支持DT_INT8、DT_INT16、DT_INT32、DT_INT64、DT_UINT8、DT_UINT16、DT_UINT32、DT_UINT64、DT_FP16、DT_BF16、DT_FP32；DT_BOOL操作数按DT_INT64参与计算（取整结果为自身），无符号整数操作数保持不变。 |

## 返回值说明

返回`ceil(x)`，为标量值表达式：整数操作数返回原值，浮点操作数返回向上取整结果（类型与操作数一致）；x为DT_BOOL时返回DT_INT64标量。

## 约束说明

- **适用范围**：本接口仅用于标量向上取整，常用于循环边界、索引计算；针对Tile逐元素向上取整请使用[pypto.ceil](../../../tensor_api/operation/pypto-ceil.md)。

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    # x 为 Kernel 内的浮点标量表达式，结果为同类型浮点标量
    rounded = pl.ceil(x)

    # 循环边界使用整数向上取整除法，不能用浮点结果
    n_blocks = pl.ceildiv(total, block_size)
    for i in pl.range(n_blocks):
        ...
```
