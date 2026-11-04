# pypto_pro.language.astype

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

将标量表达式的结果转换为指定数据类型。由于部分接口只接受特定dtype的标量，以及部分接口需要相同dtype的标量，我们需要对标量进行类型转换。
普通标量算术、比较和min/max支持整数与浮点数的隐式类型提升：混用时，两侧操作数转换为DT_FP32后计算。整数之间的`/`同样自动转换为DT_FP32，`//`用于整数整除。

## 函数原型

```python
result = pypto_pro.language.astype(x, dtype, /)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                     |
| ------ | --------- | ---------------------------------------- |
| x      | 输入      | 源标量表达式，Scalar类型，按位置传递。   |
| dtype  | 输入      | 目标数据类型，DataType类型，按位置传递。 |

## 约束说明

- 源标量和目标标量数据类型均不支持仅用于存储的低精度类型：DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_FP4、DT_FP4E2M1、DT_FP4E1M2、DT_HF8、DT_HF4、DT_INT4、DT_UINT4。
- 普通Scalar支持DT_BF16与BOOL、整数、DT_FP16、DT_FP32之间的转换。转换为DT_BF16时采用最近偶数舍入；
  DT_BF16转换为整数时向零截断，输入须位于目标整数类型的有效范围内。
- 普通Scalar的运行期DT_BF16算术先将操作数精确扩展为DT_FP32，再将每个操作的结果舍入为DT_BF16。
- 当前前端常量折叠使用Python浮点精度，尚未按FP16/BF16/FP32逐步舍入，因此常量表达式与运行期表达式可能不同。
  例如BF16常量表达式`1.0 % 0.1`折叠后约为`0.1`，而相同输入的运行期BF16计算结果为`0.09912109375`。

## 返回值说明

返回数据类型为dtype的Scalar。

## 调用示例

```python
import pypto_pro.language as pl

# 将INT32标量转换为INT64
source = pl.const(1, pl.DT_INT32)
converted = pl.astype(source, pl.DT_INT64)

# 显式统一控制流两个分支的标量类型
if flag:
    value = pl.astype(lhs, pl.DT_INT64)
else:
    value = pl.astype(rhs, pl.DT_INT64)
```
