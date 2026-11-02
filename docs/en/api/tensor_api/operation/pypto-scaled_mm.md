# pypto.scaled\_mm

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:26:40.924Z pushedAt=2026-10-28T11:57:30.078Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Implements the mx quantized matrix multiplication of the **mat_a** and **mat_b** matrices. The computation formula is: out = (mat_a \* scale_a) @ (mat_b \* scale_b)

- **mat_a**, **mat_b**, **scale_a**, and **scale_b** are source operands. **mat_a** is the left matrix, **mat_b** is the right matrix, **scale_a** is the quantization parameter of the left matrix, and **scale_b** is the quantization parameter of the right matrix.
- **out** is the destination operand, which is the matrix that stores the matrix multiplication result.

## Prototype

```python
scaled_mm(mat_a, mat_b, out_dtype, scale_a, scale_b, *, a_trans = False, b_trans = False, scale_a_trans = False, scale_b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters

Table 1: API Parameters

| Parameter | Input/Output | Description |
|-------------------|-----------|----------------------------------------------------------------------|
| mat_a | Input | Indicates the input left matrix. Does not support input of an empty tensor.<br> **Data Type**: See Table 3.<br> **Matrix Dimension**: 2D, 3D, or 4D.<br> **Format**: TILEOP_ND, TILEOP_NZ (DT_FP8E5M2 input does not support the TILEOP_NZ format).<br> **Inner Axis and Outer Axis**: When the input matrix mat_a is not transposed, the corresponding data is arranged as [M, K], where the outer axis is M and the inner axis is K. When the input matrix mat_a is transposed, the corresponding data is arranged as [K, M], where the outer axis is K and the inner axis is M.<br> **Alignment Requirement**: When Format is TILEOP_ND (ND format), the outer axis ranges from [1, 2^31 - 1] and the inner axis ranges from [1, 65535].<br> When Format is TILEOP_NZ (NZ format), the Shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis.<br> When using the pypto.view API, ensure that the Shape dimension passed to View also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| mat_b | Input | Indicates the input right matrix. Does not support input of an empty tensor.<br> **Data Type**: See Table 3.<br> **Matrix Dimension**: 2D, 3D, or 4D.<br> **Format**: TILEOP_ND, TILEOP_NZ (DT_FP8E5M2 input does not support the TILEOP_NZ format).<br> **Inner Axis and Outer Axis**: When the input matrix mat_b is not transposed, the corresponding data is arranged as [K, N], where the outer axis is K and the inner axis is N. When the input matrix mat_b is transposed, the corresponding data is arranged as [N, K], where the outer axis is N and the inner axis is K.<br> **Alignment Requirement**: When Format is TILEOP_ND (ND format), the outer axis ranges from [1, 2^31 - 1] and the inner axis ranges from [1, 65535].<br> When Format is TILEOP_NZ (NZ format), the Shape dimension must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis.<br> When using the pypto.view API, ensure that the Shape dimension passed to View also satisfies 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| out_dtype | Output | Output matrix data type. For the supported output data types in the basic scenario, see Table 3. For the supported output data types in the quantization scenario, see Table 4. |
| scale_a | Input | Indicates the quantization parameter of the input left matrix. Does not support input of an empty tensor.<br> **Data Type**: See Table 3.<br> **Quantization Parameter Dimension**: One more dimension than mat_a. When mat_a is 2D/3D/4D, scale_a is 3D/4D/5D, respectively.<br> **Format**: TILEOP_ND.<br> **Quantization Parameter Shape**: Let Ks = CeilAlign(K, 64)/64. In the 2D scenario, the non-transposed shape is [M, Ks, 2] and the transposed shape is [Ks, M, 2]. In the 3D and 4D scenarios, the Batch prefix of mat_a is prepended to the above shape. Each Batch axis of scale_a must be strictly equal to that of mat_a and use the same broadcast index as mat_a. |
| scale_b | Input | Indicates the quantization parameter of the input right matrix. Does not support input of an empty tensor.<br> **Data Type**: See Table 3.<br> **Quantization Parameter Dimension**: One more dimension than mat_b. When mat_b is 2D/3D/4D, scale_b is 3D/4D/5D, respectively.<br> **Format**: TILEOP_ND.<br> **Quantization Parameter Shape**: Let Ks = CeilAlign(K, 64)/64. In the 2D scenario, the non-transposed shape is [Ks, N, 2] and the transposed shape is [N, Ks, 2]. In the 3D and 4D scenarios, the Batch prefix of mat_b is prepended to the above shape. Each Batch axis of scale_b must be strictly equal to that of mat_b and use the same broadcast index as mat_b. |
| a_trans | Input | Whether the left input matrix is transposed. Defaults to **False**. |
| b_trans | Input | Whether the right input matrix is transposed. Defaults to **False**. |
| scale_a_trans | Input | Whether the quantization parameter of the left input matrix is transposed. Defaults to **False**. |
| scale_b_trans | Input | Whether the quantization parameter of the right input matrix is transposed. Defaults to **False**. |
| c_matrix_nz | Input | Whether the output matrix uses the NZ format. Defaults to **False**. Currently only **False** is supported, meaning the output matrix supports only the ND format. |
| extend_params | Input | Supports the quantization functions of bias and fixpipe. The data type is a dictionary. Defaults to **None**. See Table 2. |

where:
    - The element alignment of `CeilAlign(value, align)` is implemented as `(value + align - 1) / align * align`

Table 2: extend_params Parameter Description

| Parameter | Description |
|-------------------|----------------------------------------------------------------------|
| scale | Quantization parameter of the output matrix in the pertensor quantization scenario (using the same scaling factor to map high-precision numbers to low-precision numbers).<br>The input is of the float type, with 1 sign bit + 8 exponent bits + 10 mantissa bits participating in the operation.<br>For the supported input and output data types, see Table 4.<br>Multi-core K-splitting is not supported. |
| scale_tensor | Indicates the matrix for output matrix quantization in the perchannel quantization scenario (a set of quantization parameters is computed independently for each output channel).<br> The scale_tensor input is fixed as a uint64_t or int64_t tensor. During computation, the lower 32 bits of the 64-bit value are converted to the float type, and then 1 sign bit + 8 exponent bits + 10 mantissa bits are used in the operation.<br> For the supported input and output data types, see Table 4.<br> The first dimension of scale_tensor must be set to 1, and the N dimension must be equal to the N dimension of the mat_b matrix.<br> scale_tensor supports only the ND format.<br> Only 2D matrix scenarios are supported.<br> Multi-core K-splitting is not supported.<br> When the quantized output type is DT_INT8, call torch_npu.npu_trans_quant_param outside the function in advance and pass a float32 torch.tensor to obtain the int64 scale_tensor. |
| bias_tensor | Bias matrix.<br>The input is of the Tensor type.<br>The bias matrix data type can be DT_FP16, DT_BF16, or DT_FP32.<br>**bias_tensor** supports only the ND format.<br>Only 2D/3D/4D matrix scenarios are supported.<br>When the input matrix is 3D, the bias dimension can be [B, 1, N] or [1, N], and the N dimension must be equal to the N dimension of the **mat_b** matrix.<br>When the input matrix is 4D, the bias dimension can only be [1, N], and the N dimension must be equal to the N dimension of the **mat_b** matrix.<br>Multi-core K-splitting is not supported. |
| relu_type | Whether to perform the ReLu operation on the output matrix.<br>The input is of the [ReLuType](../datatype/ReLuType.md) type.<br>Supports the RELU and NO_RELU modes.<br>Multi-core K-splitting is not supported. |

Table 3: Data Types Supported by scaled_mm in the Basic Scenario

| mat_a | mat_b | out_dtype | scale_a | scale_b | bias_tensor |
|:------|:------|:----------|:--------|:--------|:------------|
| DT_FP8E5M2 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP8E4M3 | DT_FP8E5M2/DT_FP8E4M3 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_FP16/DT_BF16/DT_FP32 | DT_FP8E8M0 | DT_FP8E8M0 | DT_FP16/DT_BF16/DT_FP32 |

Table 4: Data types supported by scaled_mm in quantization scenarios

| mat_a | mat_b | out_dtype |
|:------|:-----|:----------|
| DT_FP8E5M2 | DT_FP8E5M2, DT_FP8E4M3 | DT_INT8 |
| DT_FP8E4M3 | DT_FP8E5M2, DT_FP8E4M3 | DT_INT8 |
| DT_FP4_E2M1 | DT_FP4_E2M1 | DT_INT8 |

## Return Value

Returns the out matrix, which is a tensor.

## Constraints

- When the input is in the DT_FP4_E2M1 quantization scenario, ensure that the inner axis is even.
- Before calling the scaled_mm API, use pypto.set\_cube\_tile\_shapes to set the splitting sizes on the M, K, and N axes.
- When the input to the `scaled_mm` API is in NZ format after `pypto.reshape`, you must call `pypto.set_matrix_size` to specify the original `m`, `k`, and `n` dimensions of the matrix multiplication, that is, the shape before `pypto.reshape` is applied.

## Examples

```python
# Basic matrix multiplication
mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([64, 2, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# 3D one-sided broadcast: A/scale_a is broadcast as a whole along the Batch axis.
mat_a = pypto.tensor([1, 128, 256], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([4, 256, 64], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([1, 128, 4, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([4, 4, 64, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# 4D cross broadcast: the scale Batch prefix stays consistent with the paired matrix.
mat_a = pypto.tensor([2, 1, 128, 256], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([1, 3, 256, 64], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 1, 128, 4, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([1, 3, 4, 64, 2], pypto.DT_FP8E8M0, "scale_b")
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b)

# Add Bias
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
bias = pypto.tensor((1, 32), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

# Quantization with RELU
scale_cpu = pypto.tensor((1, 32), pypto.DT_UINT64, "tensor_scale")
scale_tensor = torch_npu.npu_trans_quant_param(scale_cpu.npu()) # Generate scale_tensor.
mat_a = pypto.tensor([128, 64], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([32, 128], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
out = pypto.scaled_mm(mat_a, mat_b, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True, extend_params=extend_params)

```
