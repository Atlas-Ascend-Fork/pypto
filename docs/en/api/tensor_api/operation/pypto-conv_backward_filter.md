# pypto.conv_backward_filter

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

## Description

Computes the convolution backward filter gradient. Based on the output gradient grad_output and the input feature map of the forward convolution, it computes the gradient of the convolution kernel (weight).

## Prototype

```python
conv_backward_filter(grad_output, input, filter_size, out_dtype, strides, paddings, dilations, *, groups=1) -> Tensor
```

## Parameters

| Parameter        | Input/Output| Description                                                                |
|----------------|-----------|----------------------------------------------------------------------|
| grad_output    | Input     | Output gradient tensor.<br>Empty tensors are not supported.<br>Only 4D (2D conv) is supported, and only the NCHW format is supported.<br>Supported data types: DT_FP16 and DT_BF16.<br>Shape constraints: The value range of each dimension is [1, 1000000]. Cout must be consistent with Cout of filter_size.|
| input          | Input     | Input feature map tensor of forward convolution.<br>Only 4D (2D conv) is supported, and only the NCHW format is supported.<br>The data type must be consistent with that of grad_output.<br>Shape constraints: The value range of each dimension is [1, 1000000]. Cin must be equal to Cin of filter_size multiplied by groups. Batch must be consistent with that of grad_output.|
| filter_size    | Input     | Shape of the convolution kernel, which is also the shape of the output gradient tensor. The format is (Cout, Cin, Kh, Kw), where Cin indicates the number of input channels in each group.<br>Kh and Kw must be within [1, 255], and Kh × Kw × 32bytes/sizeof(dtype) ≤ 65535 must be satisfied.|
| out_dtype      | Input     | Data type of the output tensor.<br>Supported types: DT_FP16 and DT_BF16. The data type must be consistent with that of grad_output.|
| strides        | Input     | Convolution stride. Example: [1, 1].<br>Value range of each dimension: [1, 63].|
| paddings       | Input     | Convolution padding. Example: [1, 1, 2, 2].<br>Value range of each dimension: [0, 255], and paddingH < (Kh-1) * dilationH + 1 and paddingW < (Kw-1) * dilationW + 1 must be satisfied.|
| dilations      | Input     | Dilation rate of the dilated convolution. Example: [1, 1].<br>Value range of each dimension: [1, 63].|
| groups         | Input     | Number of groups for group convolution. The default value is 1.<br>Currently, only 1 is supported.|

## Return Value

Returns the output tensor after the convolution backward filter gradient computation:

2D convolution backward output shape: (Cout, Cin, Kh, Kw), which is the same as filter_size.

The value range of each dimension of the output shape: [1, 1000000].

## Constraints

- Before calling the conv_backward_filter API, you must call the pypto.set_convbp_filter_tile_shapes API to set the convolution backward filter gradient TileShape tiling sizes at the L1/L0 levels, and call the pypto.set_vec_tile_shapes API to set the TileShape tiling size for each dimension of the input shape.

- Only 2D convolution is supported. This means both grad_output and input are 4D, and the length of filter_size is 4.

- Hout and Wout of grad_output must meet the convolution relationship with input, strides, paddings, and dilations: Hout = (Hin + paddingTop + paddingBottom - dilationH * (Kh - 1) - 1) / strideH + 1. The same applies to Wout.

- Dynamic shapes are not supported.

## Examples

```python
# Basic example of 2D convolution backward filter gradient computation
grad_output = pypto.tensor((1, 16, 8, 8), pypto.DT_FP16, "grad_output")
input = pypto.tensor((1, 16, 8, 8), pypto.DT_FP16, "input")

tile_l1_info = pypto.pypto_impl.ConvBpTileL1Info(
        tileML1=16, tileKL1=32, tileNL1=144
    )
tile_l0_info = pypto.pypto_impl.ConvBpTileL0Info(
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
