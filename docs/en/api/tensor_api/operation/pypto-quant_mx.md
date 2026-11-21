# pypto.quant_mx

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

Quantizes a 1D to 4D high-precision floating-point tensor in ND format to a tensor in MX (Microscaling) format and returns the quantization result and shared exponent scale.

- The input tensor supports DT_FP16, DT_BF16, and DT_FP32.
- The output quantized tensor supports DT_FP8E4M3 and DT_FP4_E2M1. DT_FP4_E2M1 supports only DT_FP16 and DT_BF16 inputs.
- The data type of the scale tensor is fixed to DT_FP8E8M0.
- Quantization can be performed along the last axis (`axis=-1`) or the second-to-last axis (`axis=-2`). The ROUND_DOWN (OCP) and ROUND_UP (NV) modes are supported.
- `axis=-1` supports both the performance mode and non-performance mode. The non-performance mode supports more flexible view shape and TileShape settings, which facilitates operator fusion but reduces the performance of a single operator.

If the input shape is denoted as $[d_0, d_1, ..., d_{n-1}]$, then:

- The shape of the quantization result `quantized` is the same as that of `input`.
- FP4 uses the logical element shape: Two elements of `DT_FP4_E2M1` are packed into one byte. The external Torch output should use the `torch.uint8` buffer with the last axis halved, and the kernel output parameter is declared as `DT_FP4_E2M1`. The slicing and offset within the kernel are still computed based on the complete logical shape and do not need to be divided by 2. The quantization result can be directly used as the FP4 input of `scaled_mm`.
- When `axis=-1`, the shape of the scale is $[d_0, d_1, ..., d_{n-2}, \lceil d_{n-1} / 64 \rceil, 2]$.
- When `axis=-2`, the quantization granularity is still 32. This means every 32 elements along the second-to-last axis share one exponent. Every two consecutive groups form a 64-element block. The internal raw exp shape is $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}*2]$, and the returned scale shape is $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}, 2]$, where the last dimension stores the exponents of the two 32-element groups in sequence.

## Prototype

```python
quant_mx(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_FP8E4M3,
    mode: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_DOWN,
    axis: int = -1,
    performance_mode: bool = True,
) -> Tuple[Tensor, Tensor]
```

## Parameters

| Parameter| Input/Output| Description|
|--------|-----------|------|
| input | Input| Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. Only the TILEOP_ND format is supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| quant_dtype | Input| Data type of the output tensor after quantization.<br>DT_FP8E4M3 and DT_FP4_E2M1 are supported. DT_FP4_E2M1 supports only DT_FP16 and DT_BF16 inputs.|
| mode | Input| Rounding mode of the shared exponent during quantization.<br>ROUND_DOWN (OCP) and ROUND_UP (NV) are supported.|
| axis | Input| Axis along which quantization is performed.<br>The last dimension (`-1` or `input.shape.size() - 1`) and the second-to-last axis (`-2` or `input.shape.size() - 2`) are supported.|
| performance_mode | Input| Whether to enable the performance mode.<br>Default value: `True`.|

## Return Value

Returns a 2-tuple `(quantized, scale)`.

- `quantized`: quantized tensor. The data type is specified by `quant_dtype`, and the shape is the same as that of `input`.
- `scale`: shared exponent tensor. Its data type is fixed to DT_FP8E8M0. When `axis=-1`, the shape is `[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`; when `axis=-2`, the shape is `[*input.shape[:-2], input.shape[-2] / 64, input.shape[-1], 2]`.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
2. If TileShape is set, its dimensions must be the same as those of the input.
3. When `axis=-1` and `performance_mode=True` are true, the last axis of the view shape must be 256-byte aligned and evenly divide the last axis of the runtime shape, and the last axis of the TileShape must be equal to that of the view shape.
4. When `axis=-1` and `performance_mode=False`, the last axis of the input must be a multiple of 64. The last axis of the TileShape must be a positive number and does not need to equal the last axis of the view shape or be 256-byte aligned.
5. `axis=-2` requires that the input be at least two-dimensional. The second-to-last axis of the view shape, the runtime effective shape, and TileShape must be a positive number and a multiple of 64; the tail block of the second-to-last axis is not supported.
6. When `axis=-2` and `quant_dtype=DT_FP8E4M3`, the last axes of the view shape and TileShape must be positive numbers, and the last axis of TileShape does not need to be 256-byte aligned or equal that of the view shape.
7. When `axis=-2` and `quant_dtype=DT_FP4_E2M1`, the last axis of the view shape and TileShape must be a multiple of 64; this means each row after packing must be 32-byte aligned.
8. When `axis=-2`, the constraints on `performance_mode=True` and `performance_mode=False` are the same.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

For details about the value requirements of TileShape, see [Constraints](#constraints).

Example 1: If `axis=-1` and `performance_mode=True`, the input view shape is `[m, n]`, and TileShape can be set to `[m1, n]`.

```python
pypto.set_vec_tile_shapes(4, 64)
```

Example 2: If `axis=-1` and `performance_mode=False`, the input shape is `[m, n]`, and TileShape can be set to `[m1, n1]`.

```python
pypto.set_vec_tile_shapes(2, 128)
```

### API Call Example

```python
x = pypto.tensor([8, 64], pypto.DT_FP32)

# Default configuration: DT_FP8E4M3 + ROUND_DOWN + quantization of the last dimension
quantized, scale = pypto.quant_mx(x)

# Explicitly specify the OCP parameters.
quantized_perf, scale_perf = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -1,
    True,
)

# Use the NV scale algorithm.
quantized_nv, scale_nv = pypto.quant_mx(
    x,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    True,
)

# Disable the performance mode.
x_non_perf = pypto.tensor([2, 512], pypto.DT_FP32)
quantized_general, scale_general = pypto.quant_mx(
    x_non_perf,
    pypto.DT_FP8E4M3,
    pypto.ROUND_UP,
    -1,
    False,
)

# Quantize the second-to-last axis.
x_axis2 = pypto.tensor([128, 96], pypto.DT_FP16)
pypto.set_vec_tile_shapes(64, 96)
quantized_axis2, scale_axis2 = pypto.quant_mx(
    x_axis2,
    pypto.DT_FP8E4M3,
    pypto.ROUND_DOWN,
    -2,
    False,
)
# quantized_axis2.shape is [128, 96], and scale_axis2.shape is [2, 96, 2].
```

Result example:

```python
Input x.shape: [8, 64]
Input x.dtype: DT_FP32
Output quantized.shape: [8, 64]
Output quantized.dtype: DT_FP8E4M3
Output scale.shape: [8, 1, 2]
Output scale.dtype: DT_FP8E8M0
```
