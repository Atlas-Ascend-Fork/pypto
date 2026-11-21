# pypto.conv

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
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
<!-- npu="9050" id12 -->
- Kirin 9050 processors: Not supported
<!-- end id12 -->


## Description

Performs convolution on the input (input_conv) and weight, and supports the bias parameter. The computation formula is: out = input_conv @ weight + bias (@ denotes convolution processing).

- input_conv, weight, and bias are source operands. input_conv is the input matrix, weight is the weight matrix, and bias is the input bias.
- out is the destination operand, which is the matrix that stores the convolution result.
- Currently, the fixpipe quantization scenario is not supported.

## Prototype

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## Parameters

| Parameter           | Input/Output| Description                                                                |
|-------------------|-----------|----------------------------------------------------------------------|
| input_conv       | Input     | Input feature map tensor.<br>Empty tensors are not supported.<br>Supported dimensions: 3D (1D conv), 4D (2D conv), and 5D (3D conv).<br>Supported formats: NCL, NCHW, and NCDHW.<br>Supported data types: DT_FP16, DT_BF16, and DT_FP32.<br>Value range of each dimension: [1, 1000000].<br>The Cin of input_conv must satisfy: Cin of weight * groups = Cin of input_conv.|
| weight            | Input     | Convolution kernel tensor.<br>The dimension must be consistent with that defined by input_conv (3D/4D/5D).<br>The data type must be consistent with that defined by input_conv.<br>Value ranges of Cout, Cin, and Kd: [1, 1000000].<br>Value ranges of Kh and Kw: [1, 255]. Kh × Kw × (32 / sizeof(dtype)) ≤ 65535 must be satisfied, where dtype is the number of bytes occupied by the data type defined by input_conv, for example, 2 for FP16 and 4 for FP32.|
| out_dtype         | Input     | Data type of the output tensor.<br>Supported types: DT_FP16, DT_BF16, and DT_FP32.<br>The value must be consistent with that defined by input_conv. In the fixpipe quantization scenario, this parameter can be specified separately.|
| strides           | Input     | Convolution stride, a unidirectional parameter.<br>- 1D (1D conv)<br>- 2D (2D conv)<br>- 3D (3D conv)<br>Value range: [1, 63].|
| paddings          | Input     | Convolution padding, a bidirectional parameter.<br>- 2D (1D conv)<br>- 4D (2D conv)<br>- 6D (3D conv)<br>Value range: [0, 255]. The padding value of each dimension must be less than the corresponding convolution kernel size.|
| dilations         | Input     | Dilation rate of the dilated convolution, a unidirectional parameter.<br>- 1D (1D conv)<br>- 2D (2D conv)<br>- 3D (3D conv)<br>Value range: [1, 63].|
| groups            | Input     | Number of groups for group convolution. The default value is 1.<br>Value range: [1, 65535].<br>Cin and Cout must be exactly divisible by groups.|
| transposed        | Input     | Whether to perform a transposed convolution (deconvolution). The default value is False.<br>Currently, True is not supported.|
| output_paddings   | Input     | Output padding of transposed convolution, used only when transposed is True.<br>Not supported.|
| extend_params     | Input     | Extended parameter dictionary, supporting bias_tensor, scale, relu_type, and scale_tensor:<br>- bias_tensor: optional bias tensor with shape (Cout,), supporting only the ND format. The supported data types vary by model. For details, see [Constraints](#constraints).<br>- scale: floating-point type, per-tensor scaling factor.<br>- scale_tensor: uint64 type, per-channel scaling tensor with shape [1, Cout], supporting only the ND format.<br>- relu_type: activation type, supporting RELU/NO_RELU and others.|

## Return Value

Returns the output tensor after the convolution operation:

- 1D convolution output shape: (Batch, Cout, Wout)
- 2D convolution output shape: (Batch, Cout, Hout, Wout)
- 3D convolution output shape: (Batch, Cout, Dout, Hout, Wout)

All dimensions of the output shape support dynamic axis tiling. Note that tiling Cout is not allowed when groups is greater than 1. The value range of each dimension of the output shape is [1, 1000000].

## Constraints

1. Cache space constraint: Before calling the conv API, you must use the pypto.set_conv_tile_shapes API to set the convolution TileShape tiling size at the L1/L0 level.

2. Constraints on dynamic axis tiling:

    <!-- npu="950" id10 -->
    - Ascend 950PR&950DT products: During frontend loop tiling, the Batch, Cout, Dout, Hout, and Wout dimensions must be less than or equal to the corresponding dimension size of the out shape. Frontend loop tiling of Cin is not supported.
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    - Atlas A3 products: Frontend loop tiling of Cin is not supported.
    <!-- end id11 -->
    <!-- npu="910b" id6 -->
    - Atlas A2 products: Frontend loop tiling of Cin is not supported.
    <!-- end id6 -->

3. Data type constraints:

    <!-- npu="950" id7 -->
    - Ascend 950PR&950DT products: The supported data types are DT_FP16, DT_BF16, and DT_FP32. The data types of input_conv, weight, bias, and out must be the same.
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - Atlas A3 products: The supported data types are DT_FP16, DT_BF16, and DT_FP32. For DT_FP16 and DT_FP32, the data types of input_conv, weight, bias, and out must be the same. For DT_BF16, input_conv, weight, and out must be of type DT_BF16, and bias must be of type DT_FP32.
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    - Atlas A2 products: The supported data types are DT_FP16, DT_BF16, and DT_FP32. For DT_FP16 and DT_FP32, the data types of input_conv, weight, bias, and out must be the same. For DT_BF16, input_conv, weight, and out must be of type DT_BF16, and bias must be of type DT_FP32.
    <!-- end id9 -->

## Examples

```python
# Basic 2D convolution example
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D convolution with bias and ReLU
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias, 'relu_type': pypto.ConvReLuType.RELU}

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D convolution example
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```
