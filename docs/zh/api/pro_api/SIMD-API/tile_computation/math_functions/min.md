# pypto_pro.language.min

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

取两个标量中的较小值。

## 函数原型

```python
pypto_pro.language.min(lhs: Union[int, float], rhs: Union[int, float], /) -> Scalar
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| lhs | 输入 | 左操作数，为整型或浮点型常量，或运行时整型或浮点型标量表达式。 |
| rhs | 输入 | 右操作数，为整型或浮点型常量，或运行时整型或浮点型标量表达式。 |

## 返回值说明

返回两个标量中的较小值，Scalar类型。返回值数据类型由参数的数据类型提升结果决定。

## 约束说明

- lhs和rhs必须是数值标量。两个操作数同为整型或同为浮点型时，结果提升为位宽较大的数据类型；一个为整型、另一个为浮点型时，两个操作数均转换为FP32后再比较。
- 接口只接受两个位置参数，不支持关键字参数。
- 适用范围：本接口仅用于标量取较小值。Tile逐元素取最小值请使用pypto_pro.language.minimum。

## 调用示例

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scalar_min_kernel(
    lhs: pl.DT_FP32,
    rhs: pl.DT_FP32,
    out: pl.Tensor[[1], pl.DT_FP32],
):
    value = pl.min(lhs, rhs)
    pl.setval(out, 0, value)
```

例如lhs=2.5、rhs=-1.5时，out[0]为-1.5。整型标量可按相同方式在Kernel中用于循环边界或尾块大小计算。
