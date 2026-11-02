# pypto.quant_mx

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:18:30.959Z pushedAt=2026-10-28T11:57:30.046Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not Supported
<!-- end id3 -->

## Description

Quantizes a 1- to 4-dimensional ND-format high-precision floating-point tensor into the MX (Microscaling) format, and returns the quantization result and the shared exponent scale.

- The input tensor supports DT_FP16, DT_BF16, and DT_FP32.
- The output quantized tensor supports DT_FP8E4M3 and DT_FP4_E2M1X2. DT_FP4_E2M1X2 only supports DT_FP16 and DT_BF16 inputs.
- The data type of the scale tensor is fixed to DT_FP8E8M0.
- Supports quantization on the last axis (`axis=-1`) or the second-to-last axis (`axis=-2`), and supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes.
- `axis=-1` supports both performance mode and non-performance mode. Non-performance mode supports more flexible view shape and TileShape settings, which facilitates operator fusion, but the single-operator performance is reduced.

If the input shape is denoted as $[d_0, d_1, ..., d_{n-1}]$, then:

- The shape of the quantization result `quantized` is the same as that of `input`.
- When `axis=-1`, the shape of scale is $[d_0, d_1, ..., d_{n-2}, \lceil d_{n-1} / 64 \rceil, 2]$.
- When `axis=-2`, the quantization granularity is still 32, that is, every 32 second-to-last axis elements share one exponent; every two consecutive groups form a 64-element block. The internal raw exp shape is $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}*2]$, and the returned scale shape is $[d_0, d_1, ..., d_{n-3}, d_{n-2}/64, d_{n-1}, 2]$, where the last dimension stores the exponents of the two 32-element groups in sequence.

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

| Parameter | Input/Output | Description |
|--------|-----------|------|
| input | Input | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, DT_FP32.<br>Only supported in TILEOP_ND format; Shape supports 1 to 4 dimensions only. |
| quant_dtype | Input | Data type of the quantized output tensor.<br>Supported: DT_FP8E4M3 and DT_FP4_E2M1X2. DT_FP4_E2M1X2 only supports DT_FP16 and DT_BF16 inputs. |
| mode | Input | Rounding mode of the shared exponent during quantization.<br>Supported: ROUND_DOWN (OCP) and ROUND_UP (NV). |
| axis | Input | Specifies the quantization axis.<br>Supports the last dimension (`-1` or `input.shape.size() - 1`) and the second last axis (`-2` or `input.shape.size() - 2`). |
| performance_mode | Input | Whether to enable performance mode.<br>Default value: `True`. |

## Return Value

Returns a two-tuple `(quantized, scale)`:

- `quantized`: Quantized tensor, whose data type is specified by `quant_dtype` and whose shape is the same as `input`.t
- `scale`: Shared exponent Tensor, with a fixed data type of DT_FP8E8M0. When `axis=-1`, the shape is `[*input.shape[:-1], ceil(input.shape[-1] / 64), 2]`; when `axis=-2`, the shape is `[*input.shape[:-2], input.shape[-2] / 64, input.shape[-1], 2]`.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
2. If TileShape is set, its dimensions must be consistent with the input.
3. When `axis=-1` and `performance_mode=True`, the last axis of the view shape must satisfy 256-byte alignment and evenly divide the last axis of the runtime shape, and the last axis of TileShape must be equal to the last axis of the view shape.
4. When `axis=-1` and `performance_mode=False`, the last axis of the input must be a multiple of 64; the last axis of TileShape only needs to be a positive number and does not need to equal the last axis of the view shape or satisfy 256-byte alignment.
5. `axis=-2` requires the input to be at least two-dimensional. The second-to-last axis of the view shape, the runtime effective shape, and TileShape must be a positive number and a multiple of 64; the tail block of the second-to-last axis is not supported.
6. When `axis=-2` and `quant_dtype=DT_FP8E4M3`, the last axes of the view shape and TileShape only need to be positive numbers, 256-byte alignment is not required, and the last axis of TileShape does not need to equal the last axis of the view shape.
7. When `axis=-2` and `quant_dtype=DT_FP4_E2M1X2`, the last axis of the view shape and TileShape must be a multiple of 64, that is, each row satisfies 32-byte alignment after packing.
8. When `axis=-2`, the constraints for `performance_mode=True` and `performance_mode=False` are the same.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

For the specific value requirements of TileShape, see [Constraints](#constraints).

Example 1: When `axis=-1` and `performance_mode=True`, the input view shape is `[m, n]`, and TileShape can be set to `[m1, n]`.

```python
pypto.set_vec_tile_shapes(4, 64)
```

Example 2: When `axis=-1` and `performance_mode=False`, the input shape is `[m, n]`, and TileShape can be set to `[m1, n1]`.

```python
pypto.set_vec_tile_shapes(2, 128)
```

### API Call Example

```python
x = pypto.tensor([8, 64], pypto.DT_FP32)

# Default configuration: DT_FP8E4M3 + ROUND_DOWN + quantize the last axis
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

# Disable performance mode.
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
# quantized_axis2.shape is [128, 96], scale_axis2.shape is [2, 96, 2]
```

The results are as follows:

```python
Input x.shape: [8, 64]
Input x.dtype: DT_FP32
Output quantized.shape: [8, 64]
Output quantized.dtype: DT_FP8E4M3
Output scale.shape: [8, 1, 2]
Output scale.dtype: DT_FP8E8M0
```
