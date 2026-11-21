# pypto.quant_hif4

## Applicable Products

<!-- npu="960" id1 -->
- Ascend 960DT products: Supported
<!-- end id1 -->
<!-- npu="950" id2 -->
- Ascend 950PR&950DT products: Not supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- Atlas A3 products: Not supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- Atlas A2 products: Not supported
<!-- end id4 -->
<!-- npu="9030" id5 -->
- Kirin 9030 processors: Not supported
<!-- end id5 -->
<!-- npu="x90" id6 -->
- Kirin X90 processors: Not supported
<!-- end id6 -->

## Description

Quantizes a 1D to 4D high-precision floating-point tensor in ND format to a tensor in HiF4 format and returns the quantization result and shared exponent scale.

- The input tensor supports DT_BF16.
- The output quantized tensor supports DT_HF4.
- The data type of the scale tensor is fixed to uint_32.
- Only the last axis (`axis=-1`) can be quantized.
- `scale_alg=0/1` controls whether the parameter threshold in step 4 of the quantization function is 4/2(0) or 3.75/1.875(1).

If the input shape is denoted as $[d_0, d_1, ..., d_{n-1}]$, then:

- The shape of the quantization result `output` is the same as that of `input`.
- FP4 uses the logical element shape: Two elements of `DT_HF4` are packed into one byte. The external Torch output should use the `torch.uint8` buffer with the last axis halved, and the kernel output parameter is declared as `DT_HF4`.

## Prototype

```python
def quant_hif4(
    input: Tensor,
    quant_dtype: DataType = DataType.DT_HF4,
    scale_alg: int = 0,
    axis: int = -1,
) -> Tuple[Tensor, Tensor]:
```

## Parameters

| Parameter| Input/Output| Description|
|--------|-----------|------|
| input | Input| Source operand.<br>Supported type: Tensor.<br>Supported data type of Tensor: DT_BF16.<br>Only the `TILEOP_ND` format is supported. The shape supports only 1 to 4 dimensions.|
| quant_dtype | Input| Data type of the output tensor after quantization.<br>DT_HF4 is supported.|
| scale_alg | Input| Parameter threshold in step 4 of the quantization function.<br>0: 4 and 2; 1: 3.75 and 1.875.|
| axis | Input| Axis along which quantization is performed.<br>The last dimension (`-1` or `input.shape.size() - 1`) is supported.|

## Return Value

Returns a 2-tuple `(output, scale)`.

- `output`: quantized tensor. The data type is specified by `quant_dtype`, and the shape is the same as that of `input`.
- `scale`: scaling factor. The data type is uint32. The scaling factor of every 64 quantized elements is packed into one uint32 object. The shape of `scale` is the last axis of `input` rounded up to the nearest multiple of 64. For example, the shape of the output `scale` with the input shape of [2, 7, 120] is [2, 7, 2].

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
2. If ViewShape and TileShape are set, the dimensions must be the same as those of the input, and the last axes of ViewShape and TileShape must be 64-byte aligned.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

For details about the value requirements of TileShape, see [Constraints](#constraints).

Example 1: If the input view shape is `[m, n]`, TileShape can be set to `[m1, n]`.

```python
pypto.set_vec_tile_shapes(4, 64)
```

Example 2: If the input shape is `[m, n]`, TileShape can be set to `[m1, n1]`.

```python
pypto.set_vec_tile_shapes(2, 128)
```

### API Call Example

```python
x = pypto.tensor([8, 64], pypto.DT_BF16)

# Default configuration: scale_alg = 0, last axis quantization
output, scale = pypto.quant_hif4(x)

# scale_alg explicitly specified
output_alg_1, scale_alg_1 = pypto.quant_mx(
    x,
    pypto.DT_HF4,
    1,
    -1,
)
```

Result example:

```python
Input x.shape: [8, 64]
Input x.dtype: DT_BF16
Output output.shape: [8, 64]
Output output.dtype: DT_HF4
Output scale.shape: [8, 1]
Output scale.dtype: uint32
```
