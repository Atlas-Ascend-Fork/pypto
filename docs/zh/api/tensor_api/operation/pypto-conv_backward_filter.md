# pypto.conv_backward_filter

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->

## 功能说明

实现卷积反向权重梯度（conv backward filter）计算，根据输出梯度grad_output和正向卷积的输入特征图input，计算卷积核（权重）的梯度。

## 函数原型

```python
conv_backward_filter(grad_output, input, filter_size, out_dtype, strides, paddings, dilations, *, groups=1) -> Tensor
```

## 参数说明

| 参数名         | 输入/输出 | 说明                                                                 |
|----------------|-----------|----------------------------------------------------------------------|
| grad_output    | 输入      | 输出梯度Tensor。<br>不支持空Tensor。<br>仅支持4D（2D conv），仅支持NCHW格式。<br>支持数据类型：DT_FP16、DT_BF16。<br>shape约束：各维度取值范围 [1, 1000000]，Cout必须与filter_size的Cout一致。 |
| input          | 输入      | 正向卷积的输入特征图Tensor。<br>仅支持4D（2D conv），仅支持NCHW格式。<br>数据类型必须与grad_output一致。<br>shape约束：各维度取值范围 [1, 1000000]，Cin必须等于filter_size的Cin × groups，Batch必须与grad_output一致。 |
| filter_size    | 输入      | 卷积核shape，也是输出梯度Tensor的shape，格式为(Cout, Cin, Kh, Kw)，其中Cin为每组输入通道数。<br>Kh、Kw必须在 [1, 255] 范围内，且Kh × Kw × 32bytes/sizeof(dtype) ≤ 65535。 |
| out_dtype      | 输入      | 输出Tensor数据类型。<br>支持：DT_FP16、DT_BF16，且必须与grad_output数据类型一致。 |
| strides        | 输入      | 卷积步长，参数示例：[1, 1]。<br>各维度取值范围：[1, 63]。 |
| paddings       | 输入      | 卷积填充，参数示例：[1, 1, 2, 2]。<br>各维度取值范围：[0, 255]，且paddingH < (Kh-1) * dilationH + 1、paddingW < (Kw-1) * dilationW + 1。 |
| dilations      | 输入      | 空洞卷积膨胀率，参数示例：[1, 1]。<br>各维度取值范围：[1, 63]。 |
| groups         | 输入      | 分组卷积组数，默认1。<br>当前仅支持groups = 1。 |

## 返回值说明

返回卷积反向权重梯度计算后的输出Tensor：

2D卷积反向输出shape：(Cout, Cin, Kh, Kw)，与filter_size一致。

输出shape各维度范围：[1, 1000000]。

## 约束说明

- 调用conv_backward_filter接口前，必须通过pypto.set_convbp_filter_tile_shapes接口设置L1/L0层级的卷积反向权重梯度TileShape切分大小；调用pypto.set_vec_tile_shapes接口设置输入shape每个维度TileShape切分大小。

- 仅支持2D卷积，即grad_output、input均为4D，filter_size长度为4。

- grad_output的Hout、Wout必须与input、strides、paddings、dilations满足卷积关系：Hout = (Hin + paddingTop + paddingBottom - dilationH * (Kh - 1) - 1) / strideH + 1，Wout同理。

- 不支持动态shape。

## 调用示例

```python
# 2D卷积反向求权重梯度基础示例
grad_output = pypto.tensor((1, 16, 8, 8), pypto.DT_FP16, "grad_output")
input = pypto.tensor((1, 16, 8, 8), pypto.DT_FP16, "input")

tile_l1_info = pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=32, tileNL1=144
    )
tile_l0_info = pypto_impl.ConvBpTileL0Info(
    tileML0=16, tileKL0=16, tileNL0=16
)
pypto.set_convbp_filter_tile_shapes(tile_l1_info, tile_l0_info)
pypto.set_vec_tile_shapes(16, 16, 16, 16)
out = pypto.conv_backward_filter(
    grad_output,
    input,
    [16, 16, 3, 3],
    pypto.DT_FP16,
    strides=[1, 1],
    paddings=[1, 1, 1, 1],
    dilations=[1, 1]
)
```
