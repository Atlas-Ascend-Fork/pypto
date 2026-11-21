# pypto.cast

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
<!-- npu="9030" id12 -->
- Kirin 9030 processors: Supported
<!-- end id12 -->
<!-- npu="x90" id13 -->
- Kirin X90 processors: Supported
<!-- end id13 -->
<!-- npu="9050" id21 -->
- Kirin 9050 processors: Supported
<!-- end id21 -->

## Description

Converts the precision of the source operand and destination operand tensors based on their data types. If the destination operand is an integer and the value of the source operand exceeds the data representation range of the integer, the conversion result is clamped to the maximum or minimum value of the destination operand.

## Precautions

- **The `.to()` method not supported by PyPTO tensors**: The `.to(dtype)` method is not available for PyPTO tensors. You must use `pypto.cast(tensor, dtype)` to convert the data type.

- **Details about rounding modes**: For the floating-point representation, binary rounding rules, and the specific behavior of each CastMode, see [CastMode](../datatype/CastMode.md).

## Prototype

```python
cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE,
     satmode: SaturationMode = SaturationMode.OFF) -> Tensor
```

## Parameters

| Parameter    | Input/Output| Description                                                                |
|------------|-----------|----------------------------------------------------------------------|
| input      | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_INT4, DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP4_E2M1, and DT_FP4_E1M2. <!-- npu="x90,9030,9050" id19 -->Kirin supports the following data types: DT_FP32, DT_FP16, DT_INT32, DT_INT16, DT_UINT8, and DT_INT8.<!-- end id19 --><br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dtype      | Input     | Data type after precision conversion.<br>The supported data types are DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_INT4, DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP4_E2M1, and DT_FP4_E1M2. <!-- npu="x90,9030,9050" id20 -->Kirin supports the following data types: DT_FP16, DT_INT32, DT_INT16, DT_FP32, DT_INT8, and DT_UINT8.<!-- end id20 --> |
| CastMode   | Input     | Source operand enumeration type, which is used to control the precision conversion mode. For details, see [CastMode](../datatype/CastMode.md).<br>The default value is CAST_NONE. The framework automatically converts between common types, aligned with torch. For details, see the "Constraints" section.|
| SaturationMode    | Input     | Saturation mode enumeration type, which is used to control the overflow processing mode during conversion from floating-point numbers to integers. For details, see [SaturationMode](../datatype/SaturationMode.md).<br>The default value is OFF (truncation mode). When it is set to ON, values exceeding the target type range will be truncated to the maximum or minimum value (saturated truncation). For details, see the "Constraints" section.|

## Return Value

| Type| Description|
|:-----|:-----|
| Tensor | A new tensor of the dtype data type. Its value is the same as that of input.|

## Constraints

> **For details about the CastMode rounding modes, see** [CastMode](../datatype/CastMode.md).

<!-- npu="A3,910b" id4 -->

### Supported Conversions for Atlas A3 Products/Atlas A2 Products

| Source Type| Target Type| Supported CastMode| Default CastMode| Remarks|
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC, ODD| CAST_RINT | - |
| DT_FP32 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | Same-type rounding|
| DT_FP32 | DT_BF16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported.|
| DT_FP16 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_FP16 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP16 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported.|
| DT_FP16 | DT_INT8 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported.|
| DT_FP16 | DT_UINT8 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP16 | DT_INT4 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Packing type, 2 elements per byte.|
| DT_BF16 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_BF16 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_INT32 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_INT32 | DT_INT64 | Rounding modes are not supported.| - | Type extension.|
| DT_INT32 | DT_INT16 | Rounding modes are not supported.| - | Saturation control only.|
| DT_INT32 | DT_FP16 | Rounding modes are not supported.| - | In deq mode, deqscale needs to be set.|
| DT_INT16 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_INT16 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_INT64 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_INT64 | DT_INT32 | Rounding modes are not supported.| - | Saturation control only.|
| DT_UINT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT4 | DT_FP16 | Rounding modes are not supported.| - | Packing type, 2 elements per byte.|
<!-- end id4 -->

<!-- npu="950" id5 -->

### Supported Conversions for Ascend 950PR&950DT Products

The Ascend 950PR&950DT products use a different CastMode system. The internal implementation is based on template parameters such as `RoundRType`/`RoundAType`/`RoundFType`/`RoundCType`/`RoundZType`/`RoundOType`, while the user-facing API still uses the unified CastMode enum.

| Source Type| Target Type| Supported CastMode| Default CastMode| Remarks|
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | Same-type rounding (vtrc instruction).|
| DT_FP32 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC, ODD| CAST_RINT | - |
| DT_FP32 | DT_BF16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported.|
| DT_FP32 | DT_FP8E4M3 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_FP32 | DT_FP8E5M2 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_FP32 | DT_HF8 | **Only ROUND is supported.**| CAST_ROUND | H8 must use ROUND_A. Other modes are not supported.|
| DT_FP16 | DT_FP32 | Rounding modes are not supported.| - | Type extension (PART_EVEN).|
| DT_FP16 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | ROUND_PART mode.|
| DT_FP16 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported (ROUND_SAT mode).|
| DT_FP16 | DT_INT8 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | Edge cases such as inf/-inf are supported (ROUND_SAT_PART mode).|
| DT_FP16 | DT_UINT8 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | ROUND_SAT_PART mode.|
| DT_FP16 | DT_HF8 | **Only ROUND is supported.**| CAST_ROUND | H8 must use ROUND_A. Other modes are not supported.|
| DT_BF16 | DT_FP32 | Rounding modes are not supported.| - | Type extension (PART_EVEN).|
| DT_BF16 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_TRUNC | ROUND_SAT_PART mode.|
| DT_BF16 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | SAT_ROUND mode (Apply saturation before rounding).|
| DT_UINT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
| DT_UINT8 | DT_UINT16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT8 | DT_INT16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT8 | DT_INT32 | Rounding modes are not supported.| - | Type extension.|
| DT_INT16 | DT_UINT8 | Rounding modes are not supported.| - | SAT_PART mode.|
| DT_INT16 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | ROUND mode.|
| DT_INT16 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_INT16 | DT_UINT32 | Rounding modes are not supported.| - | Type extension.|
| DT_INT16 | DT_INT32 | Rounding modes are not supported.| - | Type extension.|
| DT_INT32 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | ROUND mode.|
| DT_INT32 | DT_INT16 | Rounding modes are not supported.| - | Saturation control only.|
| DT_INT32 | DT_UINT16 | Rounding modes are not supported.| - | Saturation control only.|
| DT_INT32 | DT_INT64 | Rounding modes are not supported.| - | Type extension.|
| DT_INT32 | DT_UINT8 | Rounding modes are not supported.| - | SAT_PART mode.|
| DT_INT32 | DT_FP16 | Rounding modes are not supported.| - | In deq mode, deqscale needs to be set.|
| DT_UINT32 | DT_UINT8 | Rounding modes are not supported.| - | SAT_PART mode.|
| DT_UINT32 | DT_UINT16 | Rounding modes are not supported.| - | Saturation control only.|
| DT_UINT32 | DT_INT16 | Rounding modes are not supported.| - | Saturation control only.|
| DT_INT64 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC| CAST_RINT | - |
| DT_INT64 | DT_INT32 | Rounding modes are not supported.| - | Saturation control only.|
| DT_FP8E4M3 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_FP8E5M2 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_HF8 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_BF16 | DT_FP4_E2M1 | Rounding modes are not supported.| - | - |
| DT_BF16 | DT_FP4_E1M2 | Rounding modes are not supported.| - | - |
| DT_FP4_E2M1 | DT_BF16 | Rounding modes are not supported.| - | - |
| DT_FP4_E1M2 | DT_BF16 | Rounding modes are not supported.| - | - |
<!-- end id5 -->

<!-- npu="x90,9030,9050" id14 -->
### Supported Conversions for Kirin 9030/Kirin X90/Kirin 9050 Processors

Kirin 9030/Kirin X90/Kirin 9050 processors use a different CastMode system. The internal implementation is based on template parameters such as `RoundRType`/`RoundAType`/`RoundFType`/`RoundCType`/`RoundZType`/`RoundOType`, while the user-facing API still uses the unified CastMode enum.

| Source Type| Target Type| Supported CastMode| Default CastMode| Remarks|
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC, ODD | CAST_RINT | - |
| DT_FP32 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported.|
| DT_FP16 | DT_FP32 | Rounding modes are not supported.| - | Type extension (PART_EVEN).|
| DT_FP16 | DT_INT32 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | ROUND_PART mode.|
| DT_FP16 | DT_INT16 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported (ROUND_SAT mode).|
| DT_FP16 | DT_INT8 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported (ROUND_SAT_PART mode).|
| DT_FP16 | DT_UINT8 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_TRUNC | ROUND_SAT_PART mode.|
| DT_INT32 | DT_FP32 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_RINT | ROUND mode.|
| DT_INT16 | DT_FP32 | Rounding modes are not supported.| - | Type extension.|
| DT_INT16 | DT_FP16 | RINT, ROUND, FLOOR, CEIL, TRUNC | CAST_RINT | ROUND mode.|
| DT_UINT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
| DT_INT8 | DT_FP16 | Rounding modes are not supported.| - | Type extension.|
<!-- end id14 -->

### Setting the Saturation Mode

The saturation mode (SaturationMode) controls how overflow is handled during floating-point to integer conversion:

- **OFF (default)**: truncation mode. Values that exceed the target type range are truncated in binary.
- **ON**: saturation mode. Values that exceed the target type range are clamped to the maximum or minimum value.

For scenarios such as quantization, set **`satmode=SaturationMode.ON`** to avoid precision issues caused by overflow. In other scenarios, use the default value.

---

### Other Constraints

1. When the types before and after casting are the same, a no-operation may occur in some scenarios, with no guarantee of precision.

2. **DT_INT4 (S4) special notes**: Packing type, 2 elements per byte. Only conversion to and from DT_FP16 is supported.

3. **DT_FP4_E2M1/DT_FP4_E1M2 special notes**:
    <!-- npu="950" id6 -->
    - Ascend 950PR&950DT products: Supported
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - Atlas A3 products: Not supported
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2 products: Not supported
    <!-- end id8 -->
    <!-- npu="9030" id15 -->
    - Kirin 9030 processors: Not supported
    <!-- end id15 -->
    <!-- npu="x90" id16 -->
    - Kirin X90 processors: Not supported
    <!-- end id16 -->
    <!-- npu="9050" id22 -->
    - Kirin 9050 processors: Not supported
    <!-- end id22 -->
    - The logic 4-bit type is used, which is consistent with that of scaled_mm. The input parameter annotation and cast target type use the enumeration without the X2 suffix.
    - The logical shape remains unchanged before and after conversion. Two FP4 elements are stored per byte, and the length of the logical last axis must be an even number.
    - When interacting with Torch, torch.uint8 is used to carry the packed data, and the storage shape is [m, n/2]. After the parameter is annotated as `pypto.Tensor([], pypto.DT_FP4_E1M2)` or `pypto.Tensor([], pypto.DT_FP4_E2M1)`, the logical shape seen in the kernel is [m, n].
    - Only conversion to and from DT_BF16 is supported.

4. **DT_HF8 (hifloat8) special notes**:
    <!-- npu="950" id9 -->
    - Ascend 950PR&950DT products: Supported
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - Atlas A3 products: Not supported
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2 products: Not supported
    <!-- end id11 -->
    <!-- npu="9030" id17 -->
    - Kirin 9030 processors: Not supported
    <!-- end id17 -->
    <!-- npu="x90" id18 -->
    - Kirin X90 processors: Not supported
    <!-- end id18 -->
    <!-- npu="9050" id23 -->
    - Kirin 9050 processors: Not supported
    <!-- end id23 -->
    - The CAST_ROUND rounding mode (ROUND_A of the hardware) must be used.
    - If another CastMode is specified, the implementation automatically falls back to CAST_ROUND.

5. **Unsupported CastMode handling**:
    - When you specify a CastMode that is not supported by the hardware for a conversion, the framework does not report an error.
    - The framework automatically uses the default CastMode for that conversion:
      - Floating-point to integer: CAST_TRUNC is used.
      - Other scenarios: CAST_RINT is used.

6. **deq mode description**: The conversion from INT32 to FP16 uses the deq mode, which requires setting the scaling factor through `set_deqscale`, with a default value of 1.0.
7. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2], pypto.DT_FP32)
y = pypto.cast(x, pypto.DT_FP16)
```

Result example:

```python
Input data x: [2.0, 3.0] # x.dtype: pypto.DT_FP32

Output data y: [2.0, 3.0] # y.dtype: pypto.DT_FP16
```

#### Using the Saturation Mode (Recommended for Floating-Point to Integer Conversion)

```python
# Example 1: Convert FP16 to INT8 and use the saturation mode to prevent overflow.
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
# Output: [127, -128, 50]

# Example 2: Convert FP16 to INT8 and use the saturation mode to prevent overflow.
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.OFF)
# Output: [44, -44, 50]
```
