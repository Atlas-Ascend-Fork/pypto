# pypto.matmul

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id17 -->
- Kirin 9050 processors: Supported
<!-- end id17 -->

## Description

Implements matrix multiplication of the input and mat2 matrices, with the formula: out = input @ mat2

- input and mat2 are source operands, where input is the left matrix and mat2 is the right matrix.
- out is the destination operand, which is the matrix that stores the matrix multiplication result.

## Precautions

- **Ensure the data types of the left and right matrices of matmul are the same**. For example, BF16+BF16 or FP16+FP16. Mixed inputs (for example, BF16+FP32) are not supported, except for the FP8 data type.
- **Avoid unnecessary casts**. Upgrading BF16 to FP32 before matmul computation does not improve precision; instead, it incurs additional data movement overhead.
- **Use on-the-fly transpose**. matmul supports the `a_trans` and `b_trans` parameters, which enable transpose operations to be performed on-the-fly during matrix multiplication, eliminating the need for separate transpose calls.
- **Set TileShape first**. Before calling the matmul API, you need to call `set_cube_tile_shapes` to set the tile sizes on the M, K, and N axes.

## Prototype

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters

Table 1: API parameters

| Parameter           | Input/Output| Description                                                                |
|-------------------|-----------|----------------------------------------------------------------------|
| input             | Input     | Input left matrix. Empty tensors are not supported.<br> **Data type**: For details, see Table 3.<br> **Matrix dimension**: 2D, 3D, or 4D. The dimensions of the left and right matrices must be the same.<br> **Format**: TILEOP_ND or TILEOP_NZ (DT_FP32, DT_FP8E5M2, and DT_HF8 inputs do not support the TILEOP_NZ format.)<br> **Inner/Outer axis**: When the input matrix input is not transposed, the corresponding data is arranged as [M, K], where the outer axis is M and the inner axis is K. When the input matrix input is transposed, the corresponding data is arranged as [K, M], where the outer axis is K and the inner axis is M.<br> **Alignment requirements**: When the format is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535].<br> When the format is TILEOP_NZ (NZ format), the shape dimensions must be 32-byte aligned on the inner axis and 16-element aligned on the outer axis.<br> When calling the pypto.view API, ensure that the shape dimensions of the passed View are also 32-byte aligned on the inner axis and 16-element aligned on the outer axis.|
| mat2              | Input     | Input right matrix. Empty tensors are not supported.<br> **Data type**: For details, see Table 3.<br> **Matrix dimension**: 2D, 3D, or 4D. The dimensions of the left and right matrices must be the same.<br> **Format**: TILEOP_ND or TILEOP_NZ (DT_FP32, DT_FP8E5M2, and DT_HF8 inputs do not support the TILEOP_NZ format.)<br> **Inner/Outer axis**: When the input matrix mat2 is not transposed, the corresponding data is arranged as [K, N], where the outer axis is K and the inner axis is N. When the input matrix mat2 is transposed, the corresponding data is arranged as [N, K], where the outer axis is N and the inner axis is K.<br> **Alignment requirements**: When the format is TILEOP_ND (ND format), the outer axis range is [1, 2^31 - 1] and the inner axis range is [1, 65535].<br> When the format is TILEOP_NZ (NZ format), the shape dimensions must be 32-byte aligned on the inner axis and 16-element aligned on the outer axis.<br> When calling the pypto.view API, ensure that the shape dimensions of the passed View are also 32-byte aligned on the inner axis and 16-element aligned on the outer axis.|
| out_dtype         | Output     | Output matrix data type. For details about the supported output data types in basic scenarios, see Table 3. For details about the supported output data types in dequantization and quantization scenarios, see Table 4 and Table 5.|
| a_trans           | Input     | Whether the left input matrix is transposed. The default value is False.|
| b_trans           | Input     | Whether the right input matrix is transposed. The default value is False.|
| c_matrix_nz       | Input     | Whether the output matrix uses the NZ format. The default value is False. Currently, only False is supported, that is, the output matrix supports only the ND format.|
| extend_params     | Input     | Includes bias, fixpipe quantization and dequantization, and TF32 rounding mode. For details, see Table 2.<br>For details about the supported input and output data types of bias, and fixpipe quantization and dequantization, see Table 3, Table 4, and Table 5.<br>- The data type is in dictionary format.<br>- This parameter and its internal parameters are optional.|

Table 2: Parameters in extend_params

| Parameter           | Description                                                                |
|-------------------|----------------------------------------------------------------------|
| scale             | Dequantization parameter for the output matrix in per-tensor quantization scenarios (where a single scaling factor is used to map high-precision values to low-precision values).<br>The input is of the float type, and 1 sign bit + 8 exponent bits + 10 mantissa bits are used for computation.<br>For details about the supported input and output data types, see Table 4 and Table 5.<br>Multi-core K-splitting is not supported.|
| scale_tensor      | Dequantization matrix for the output matrix in per-channel quantization scenarios (where a separate set of quantization parameters is computed independently for each output channel).<br>The scale_tensor input is a tensor of the uint64_t or int64_t type. During computation, the 64-bit value is converted to the lower 32 bits of the float type, and then 1 sign bit + 8 exponent bits + 10 mantissa bits are used for computation.<br>For details about the supported input and output data types, see Table 4 and Table 5.<br>The shape of the second-to-last dimension of scale_tensor must be set to 1, and the N dimension must be equal to the N dimension of the mat2 matrix.<br>scale_tensor supports only the ND format.<br>Multi-core K-splitting is not supported.<br>When the quantization output type is DT_INT8, call torch_npu.npu_trans_quant_param in advance and pass a float32 torch.tensor to obtain the int64 scale_tensor.|
| bias_tensor       | Bias matrix.<br>The input is of the Tensor type.<br>For details about the supported input and output data types, see Table 3.<br>bias_tensor supports only the ND format.<br>The shape of the second-to-last dimension of bias_tensor must be set to 1, and the N dimension must be equal to the N dimension of the mat2 matrix.<br>In the 4D matrix dimension scenario, bias allows only 2D inputs.<br>Multi-core K-splitting is not supported.|
| relu_type         | Whether to perform the ReLu operation on the output matrix.<br>The input is of the [ReLuType](../datatype/ReLuType.md) type.<br>The RELU and NO_RELU modes are supported.<br>Multi-core K-splitting is not supported.|
| trans_mode        | Whether to enable TF32 computation and TF32 rounding mode.<br>The input is of the [TransMode](../datatype/TransMode.md) type. The following three modes are supported:<br>• CAST_NONE: Does not enable conversion of the float data type to the TF32 data type.<br>• CAST_RINT: Enables conversion of the float data type to the TF32 data type. Rounding rule: round to the nearest integer, with ties rounding to even.<br>• CAST_ROUND: Enables conversion of the float data type to the TF32 data type. Rounding rule: round to the nearest integer, with ties rounding away from zero.<br>This parameter can be set only when the input left/right matrices and the output matrix are all of the DT_FP32 data type.|

Table 3: Data types supported in the Matmul basic scenario

| input | mat2 | out_dtype | bias_tensor |
|:------|:-----|:----------|:------------|
| DT_FP16 | DT_FP16 | DT_FP16/DT_FP32 | DT_FP16/DT_FP32 |
| DT_BF16 | DT_BF16 | DT_BF16/DT_FP32 | DT_BF16/DT_FP32 |
| DT_FP32 | DT_FP32 | DT_FP32 | DT_FP32 |
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |
| DT_HF8 | DT_HF8 | DT_FP16/DT_BF16/DT_FP32 | DT_FP16/DT_BF16/DT_FP32 |

Table 4: Data types supported in the Matmul dequantization scenario

| input | mat2 | out_dtype |
|:------|:-----|:----------|
| DT_INT8 | DT_INT8 | DT_FP16 |

Table 5: Data types supported in the Matmul quantization scenario

| input | mat2 | out_dtype |
|:------|:-----|:----------|
| DT_BF16 | DT_BF16 | DT_INT8 |
| DT_FP16 | DT_FP16 | DT_INT8 |
| DT_FP32 | DT_FP32 | DT_INT8 |
| DT_INT8 | DT_INT8 | DT_INT8 |
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_INT8 |
| DT_HF8 | DT_HF8 | DT_INT8 |

## Return Value

Returns the out matrix, which is a tensor.

## Constraints

<!-- npu="910b" id4 -->
- Atlas A2 products: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported, and the trans_mode parameter in extend_params is not supported. When the data type of the input matrix is DT_BF16, the data type of bias_tensor cannot be DT_BF16.
<!-- end id4 -->
<!-- npu="A3" id5 -->
- Atlas A3 products: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported, and the trans_mode parameter in extend_params is not supported. When the data type of the input matrix is DT_BF16, the data type of bias_tensor cannot be DT_BF16.
<!-- end id5 -->
<!-- npu="9030" id11 -->
- Kirin 9030 processors: For details about the supported data types, see [Data Types Supported by Kirin 9030 Processors](#data-types-supported-by-kirin-9030-processors). The TILEOP_NZ format inputs (input and mat2) are not supported. c_matrix_nz is not supported. scale, scale_tensor, relu_type, and trans_mode in extend_params are not supported.
<!-- end id11 -->
<!-- npu="x90" id12 -->
- Kirin X90 processors: For details about the supported data types, see [Data Types Supported by Kirin X90 Processors](#data-types-supported-by-kirin-x90-processors). The TILEOP_NZ format inputs (input and mat2) are not supported. c_matrix_nz is not supported. scale, scale_tensor, relu_type, and trans_mode in extend_params are not supported.
<!-- end id12 -->
<!-- npu="9050" id13 -->
- Kirin 9050 processors: For details about the supported data types, see [Data Types Supported by Kirin 9050 Processors](#data-types-supported-by-kirin-9050-processors). The TILEOP_NZ format inputs (input and mat2) are not supported. c_matrix_nz is not supported. `a_trans` (on-the-fly transpose of matrix A for Kirin 9030 processors/Kirin X90 processors) is not supported. scale, scale_tensor, relu_type, and trans_mode in extend_params are not supported. To transpose matrix A, call `pypto.transpose` to manually transpose `input`, and then call `matmul` with `a_trans=False`, for example, `a_t = pypto.transpose(a, -1, -2)`, `pypto.matmul(a_t, b, out_dtype, a_trans=False)`.
<!-- end id13 -->
- Before calling the matmul API, you need to set the tile size along the M, K, and N axes via pypto.set\_cube\_tile\_shapes.
- When the matrix dimension is 3D or 4D, you need to call the pypto.set\_vec\_tile\_shapes API to set the vector tile shape. If not set, the API will internally use a default 2D vector tile shape of [128, 128].
- When the input to the matmul API is in NZ format after a pypto.reshape call, you need to call the pypto.set\_matrix\_size API to set the m, k, and n values of the original shape input to the matmul API before pypto.reshape is called.
- When the input matrices to the matmul API are 3D or 4D and in NZ format, you need to call the pypto.set\_matrix\_size API to set the m, k, and n values of the original shape input to the matmul API.

<!-- npu="9030" id9 -->
### Data Types Supported by Kirin 9030 Processors
| input | mat2 | out_dtype | bias_tensor |
|:------|:-----|:----------|:------------|
| DT_FP16 | DT_FP16 | DT_FP16 | DT_FP16 |
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 |
<!-- end id9 -->

<!-- npu="x90" id10 -->
### Data Types Supported by Kirin X90 Processors
| input | mat2 | out_dtype | bias_tensor |
|:------|:-----|:----------|:------------|
| DT_FP16 | DT_FP16 | DT_FP16 | DT_FP32 |
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 |
- Kirin X90 requires that in the fp16@fp16->fp16 scenario, bias is stored as fp32 (the significant digits are actually stored as fp16), the upper 16 bits are padded with zeros (invalid), and the lower 16 bits are valid.
<!-- end id10 -->

<!-- npu="9050" id14 -->
### Data Types Supported by Kirin 9050 Processors

| input | mat2 | out_dtype | bias_tensor |
|:------|:-----|:----------|:------------|
| DT_FP16 | DT_FP16 | DT_FP16 | DT_FP16|
| DT_INT8 | DT_INT8 | DT_INT32 | DT_INT32 |

<!-- end id14 -->

## Examples

```python
# Basic matrix multiplication
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

# Batch matrix multiplication
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

# Batch broadcasting
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

# Bias addition
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)

# Dequantization
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# Dequantization with RELU
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# Dequantization with RELU
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

# Quantization
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)

# Quantization with RELU
import torch
import torch_npu

pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
scale_cpu = torch.full((1, 64), 0.2, dtype=torch.float32)
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # Generate scale_tensor.
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
pypto.matmul(a, b, pypto.DT_INT8, extend_params=extend_params)
```

<!-- npu="950" id6 -->
```python
# TF32 compute mode (Ascend 950PR&950DT products)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
a = pypto.tensor((16, 32), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP32, "tensor_b")
extend_params = {'trans_mode': pypto.TransMode.CAST_ROUND}
pypto.matmul(a, b, pypto.DT_FP32, extend_params=extend_params)
```
<!-- end id6 -->
