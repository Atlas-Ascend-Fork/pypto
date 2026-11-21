# pypto.conv_backward_input

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Computes the convolution backward input gradient. Based on the output gradient grad_output and the convolution kernel weight, it computes the gradient of the input feature map.

## Prototype

```python
conv_backward_input(grad_output, input_size, weight, out_dtype, strides, paddings, dilations, *, groups=1) -> Tensor
```

## Parameters

| Parameter        | Input/Output| Description                                                                |
|----------------|-----------|----------------------------------------------------------------------|
| grad_output    | Input     | Output gradient tensor.<br>Empty tensors are not supported.<br>Only 4D (2D conv) is supported, and only the NCHW format is supported.<br>Supported data types: DT_FP16 and DT_BF16.<br>Shape constraints: The value range of each dimension is [1, 1000000].|
| input_size     | Input     | Shape of the input feature map, which is also the shape of the output gradient. Its dimension is consistent with that of grad_output.|
| weight         | Input     | Convolution kernel tensor.<br>Its dimensions are consistent with those of grad_output (4D), and its data type must be consistent with that of grad_output.<br>Shape constraints: The value range of each dimension is [1, 1000000]; Kh and Kw must be within [1, 255], and Kh × Kw × 32bytes/sizeof(dtype) ≤ 65535.|
| out_dtype      | Input     | Data type of the output tensor.<br>Supported types: DT_FP16 and DT_BF16. The data type must be consistent with that of grad_output.|
| strides        | Input     | Convolution stride. Example: [1, 1].<br>Value range of each dimension: [1, 63].|
| paddings       | Input     | Convolution padding. Example: [1, 1, 2, 2].<br>Value range of each dimension: [0, 255], and paddingH < (Kh-1) * dilationH + 1 and paddingW < (Kw-1) * dilationW + 1 must be satisfied.<br>Currently, only symmetric padding on both sides is supported, that is, padLeft == padRight and padTop == padBottom.|
| dilations      | Input     | Dilation rate of the dilated convolution. Example: [1, 1].<br>Value range of each dimension: [1, 63].|
| groups         | Input     | Number of groups for group convolution. The default value is 1.<br>Currently, only 1 is supported.|

## Return Value

Returns the output tensor after the convolution backward input gradient computation:

2D convolution backward output shape: (Batch, Cin, Hin, Win)

The value range of each dimension of the output shape: [1, 1000000].

**Note**: The output shape cannot be uniquely derived from grad_output/weight and the attributes alone, as the shape computation involves dividing by the stride and rounding down. Therefore, the output shape is a range. The following formulas are the valid range computation formulas for each axis. Any value within [min, max] produced by the following formulas is valid, and you need to pass the desired output shape to input_size:

```txt
HinMin = (Hout - 1) * stride_h - pad_top - pad_bottom + (Kh - 1) * dilation_h + 1
WinMin = (Wout - 1) * stride_w - pad_left - pad_right + (Kw - 1) * dilation_w + 1
HinMax = HinMin + stride_h - 1
WinMax = WinMin + stride_w - 1
```

## Constraints

- Before calling the conv_backward_input API, you must call the pypto.set_convbp_input_tile_shapes API to set the convolution backward TileShape tiling sizes at the L1/L0 levels, and call the pypto.set_vec_tile_shapes API to set the TileShape tiling size for each dimension of the input shape.

- bias is not supported.

- Dynamic shapes are not supported.

## Examples

```python
# Basic example of 2D convolution backward
grad_output = pypto.tensor((1, 16, 5, 32), pypto.DT_FP16, "grad_output")
weight = pypto.tensor((16, 16, 3, 3), pypto.DT_FP16, "weight")

# Input gradient shape computation
hin = 5
win = 32
input_size = (1, 16, hin, win)
tile_l1_info = pypto.pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=144, tileNL1=16
    )
tile_l0_info = pypto.pypto_impl.ConvBpTileL0Info(
    tileML0=16, tileKL0=16, tileNL0=16
)
pypto.set_convbp_input_tile_shapes(tile_l1_info, tile_l0_info)
pypto.set_vec_tile_shapes(16, 16, 16, 16)
out = pypto.conv_backward_input(
    grad_output,
    input_size,
    weight,
    pypto.DT_FP16,
    strides=[1, 1],
    paddings=[1, 1, 1, 1],
    dilations=[1, 1]
)
```
