# pypto_pro.language.Img2ColConfig

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

[pypto_pro.language.img2col](../memory_data_movement/img2col.md)使用的卷积窗口几何描述，编译期常量配置类。描述卷积核窗口（kernel）、滑动步长（stride）、膨胀系数（dilation）、边界填充（padding）与越界填充值（pad_value），全部字段须为编译期常量，构造时即完成校验并固化进img2col调用。

通道深度不在本配置中：img2col读取的``channelSize``由源Tile的``C1 × C0``自动推导（见[img2col](../memory_data_movement/img2col.md)约束说明）。

## 原型定义

```python
pypto_pro.language.Img2ColConfig(
    *,
    kernel: Shape,
    stride: Shape,
    dilation: Shape,
    padding: Shape,
    pad_value: int = 0,
) -> Any
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|---|---|---|
| kernel | 输入 | 卷积核高宽``[Kh, Kw]``，长度为2的列表，范围[1, 255]（img2col硬件寄存器位宽）。 |
| stride | 输入 | 高宽方向滑动步长``[strideH, strideW]``，长度为2的列表，范围[1, 63]。 |
| dilation | 输入 | 高宽方向膨胀系数``[dilationH, dilationW]``，长度为2的列表，范围[1, 63]。 |
| padding | 输入 | 边界填充``[padTop, padBottom, padLeft, padRight]``，上、下、左、右顺序，长度为4的列表，范围[0, 255]。 |
| pad_value | 输入 | 越界填充值，整数，默认0。带padding时，越出特征图边界的窗口位置填充该值。 |

## 约束说明

- 全部字段必须为编译期常量：构造发生在AST解析期，不允许依赖运行时值（如tensor shape、循环变量）。
- 各字段长度必须与要求一致（kernel/stride/dilation为2、padding为4），元素必须为整数，且逐项满足上表范围。
- ``padding``顺序为上、下、左、右（用户语义）；img2col生成代码内部按硬件fmatrix寄存器的``{left, right, top, bottom}``顺序打包，无需用户关心。
- ``pad_value``为整数即可，不做类型收窄；填充行为按源Tile的dtype写入。
- 窗口几何决定目的矩阵形状（``M = align16(Ho × Wo)``、``K = C1 × Kh × Kw × C0``），推导公式见[img2col](../memory_data_movement/img2col.md)功能说明。

## 返回值说明

返回固化的``Img2ColConfig``配置对象，仅作为[pypto_pro.language.img2col](../memory_data_movement/img2col.md)的``config``入参使用。

## 调用示例

```python
import pypto_pro.language as pl

# 3x3卷积核，stride 1，无膨胀，上下左右各pad 1，越界填0（默认）
cfg = pl.Img2ColConfig(
    kernel=[3, 3], stride=[1, 1], dilation=[1, 1],
    padding=[1, 1, 1, 1],
)

# 5x5卷积核，stride 2，膨胀2
cfg = pl.Img2ColConfig(
    kernel=[5, 5], stride=[2, 2], dilation=[2, 2],
    padding=[2, 2, 2, 2], pad_value=0,
)
```
