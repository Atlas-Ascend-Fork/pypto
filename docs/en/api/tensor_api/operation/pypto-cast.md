# pypto.cast

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:13:58.481Z pushedAt=2026-10-28T11:57:29.811Z nl=2 -->

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

## Description

Performs precision conversion based on the data types of the source and destination tensors. If the destination tensor is of an integer type and the value of the source tensor exceeds the representable range of the integer type, the conversion result is clamped to the maximum or minimum value of the destination tensor.

## Precautions

- **PyPTO tensor not supporting the `.to()` method**: The PyPTO tensor does not provide a `.to(dtype)` method. You must use `pypto.cast(tensor, dtype)` for data type conversion.

- **Detailed description of rounding modes**: For the floating-point representation, binary rounding rules, and the specific behavior of each CastMode, see [CastMode](../datatype/CastMode.md).

## Prototype

```python
cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE,
     satmode: SaturationMode = SaturationMode.OFF) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|------------|-----------|----------------------------------------------------------------------|
| input | Input | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_INT4, DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP4_E2M1, and DT_FP4_E1M2.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| dtype | Input | Data type after precision conversion.<br>Supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_INT4, DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP4_E2M1, and DT_FP4_E1M2. |
| CastMode | Input | Source operand enumeration that controls the precision conversion processing mode. For details, see [CastMode](../datatype/CastMode.md).<br>Defaults to **CAST_NONE**. For conversions between common types, the framework performs the conversion automatically, aligned with **torch**. For details, see the "Constraints" section. |
| SaturationMode | Input | Saturation mode enumeration that controls the overflow handling when floating-point numbers are converted to integers. For details, see [SaturationMode](../datatype/SaturationMode.md).<br>Defaults to **OFF** (truncation mode). When set to **ON**, values exceeding the target type range are truncated to the maximum or minimum value (saturation truncation). For details, see the "Constraints" section. |

## Return Value

| Type | Description |
|:-----|:-----|
| Tensor | A new tensor whose data type is dtype and whose values are the same as **input**. |

## Constraints

> **For a detailed description of CastMode rounding modes, see** [CastMode](../datatype/CastMode.md).

<!-- npu="A3,910b" id4 -->

### Supported Conversions for Atlas A3 Products/Atlas A2 Products

| Source Type | Target Type | Supported CastMode | Default CastMode | Remarks |
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP16 | RINT,ROUND,FLOOR,CEIL,TRUNC,ODD | CAST_RINT | - |
| DT_FP32 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | Same-type rounding. |
| DT_FP32 | DT_BF16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. |
| DT_FP16 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_FP16 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP16 | DT_INT16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. |
| DT_FP16 | DT_INT8 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. |
| DT_FP16 | DT_UINT8 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP16 | DT_INT4 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Packed type, 2 elements per byte.  |
| DT_BF16 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_BF16 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_INT32 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_INT32 | DT_INT64 | Rounding mode unsupported. | - | Type extension. |
| DT_INT32 | DT_INT16 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_INT32 | DT_FP16 | Rounding mode unsupported. | - | deq mode; deqscale must be set. |
| DT_INT16 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_INT16 | DT_FP16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_INT64 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_INT64 | DT_INT32 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_UINT8 | DT_FP16 | Rounding mode unsupported. | - | Type extension. |
| DT_INT8 | DT_FP16 | Rounding mode unsupported. | - | Type extension. |
| DT_INT4 | DT_FP16 | Rounding mode unsupported. | - | Packed type, 2 elements per byte.  |
<!-- end id4 -->

<!-- npu="950" id5 -->

### Supported Conversions for Ascend 950PR&950DT Products

Ascend 950PR&950DT products use a different CastMode system. The internal implementation is based on template parameters such as `RoundRType`/`RoundAType`/`RoundFType`/`RoundCType`/`RoundZType`/`RoundOType`, while the user-facing API still uses the unified CastMode enum.

| Source Type | Target Type | Supported CastMode | Default CastMode | Remarks |
|--------|----------|----------------|--------------|----------|
| DT_FP32 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | Same-type rounding (vtrc instruction). |
| DT_FP32 | DT_FP16 | RINT,ROUND,FLOOR,CEIL,TRUNC,ODD | CAST_RINT | - |
| DT_FP32 | DT_BF16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_FP32 | DT_INT64 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | - |
| DT_FP32 | DT_INT16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. |
| DT_FP32 | DT_FP8E4M3 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_FP32 | DT_FP8E5M2 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_FP32 | DT_HF8 | **Only ROUND supported.** | CAST_ROUND | H8 must use ROUND_A; other modes are not supported. |
| DT_FP16 | DT_FP32 | Rounding mode unsupported. | - | Type extension (PART_EVEN). |
| DT_FP16 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | ROUND_PART mode. |
| DT_FP16 | DT_INT16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. ROUND_SAT mode. |
| DT_FP16 | DT_INT8 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | Edge cases such as inf/-inf are supported. ROUND_SAT_PART mode. |
| DT_FP16 | DT_UINT8 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | ROUND_SAT_PART mode. |
| DT_FP16 | DT_HF8 | **Only ROUND supported.** | CAST_ROUND | H8 must use ROUND_A; other modes are not supported. |
| DT_BF16 | DT_FP32 | Rounding mode unsupported. | - | Type extension (PART_EVEN). |
| DT_BF16 | DT_INT32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_TRUNC | ROUND_SAT_PART mode. |
| DT_BF16 | DT_FP16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | SAT_ROUND mode (saturate first, then round). |
| DT_UINT8 | DT_FP16 | Rounding mode unsupported. | - | Type extension. |
| DT_UINT8 | DT_UINT16 | Rounding mode unsupported. | - | Type extension. |
| DT_INT8 | DT_FP16 | Rounding mode unsupported. | - | Type extension. |
| DT_INT8 | DT_INT16 | Rounding mode unsupported. | - | Type extension. |
| DT_INT8 | DT_INT32 | Rounding mode unsupported. | - | Type extension. |
| DT_INT16 | DT_UINT8 | Rounding mode unsupported. | - | SAT_PART mode. |
| DT_INT16 | DT_FP16 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | ROUND mode. |
| DT_INT16 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_INT16 | DT_UINT32 | Rounding mode unsupported. | - | Type extension. |
| DT_INT16 | DT_INT32 | Rounding mode unsupported. | - | Type extension. |
| DT_INT32 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | ROUND mode. |
| DT_INT32 | DT_INT16 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_INT32 | DT_UINT16 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_INT32 | DT_INT64 | Rounding mode unsupported. | - | Type extension. |
| DT_INT32 | DT_UINT8 | Rounding mode unsupported. | - | SAT_PART mode. |
| DT_INT32 | DT_FP16 | Rounding mode unsupported. | - | deq mode. deqscale must be set. |
| DT_UINT32 | DT_UINT8 | Rounding mode unsupported. | - | SAT_PART mode. |
| DT_UINT32 | DT_UINT16 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_UINT32 | DT_INT16 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_INT64 | DT_FP32 | RINT,ROUND,FLOOR,CEIL,TRUNC | CAST_RINT | - |
| DT_INT64 | DT_INT32 | Rounding mode unsupported. | - | Saturation-only control. |
| DT_FP8E4M3 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_FP8E5M2 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_HF8 | DT_FP32 | Rounding mode unsupported. | - | Type extension. |
| DT_BF16 | DT_FP4_E2M1 | Rounding mode unsupported. | - | - |
| DT_BF16 | DT_FP4_E1M2 | Rounding mode unsupported. | - | - |
| DT_FP4_E2M1 | DT_BF16 | Rounding mode unsupported. | - | - |
| DT_FP4_E1M2 | DT_BF16 | Rounding mode unsupported. | - | - |
<!-- end id5 -->

### Setting the Saturation Mode

The saturation mode (**SaturationMode**) controls how overflow is handled during floating-point to integer conversion:

- **OFF (default)**: Truncation mode. Values that exceed the target type range are truncated in binary.
- **ON**: Saturation mode. Values that exceed the target type range are clamped to the maximum or minimum value.

For scenarios such as quantization, it is recommended to set `satmode=SaturationMode.ON` to avoid precision issues caused by overflow. For other scenarios, the default value can be used.

---

### Other Constraints

1. When the source and destination types of a cast are identical, certain scenarios may result in a no-op, with no guarantee of precision.

2. **DT_INT4 (S4) special notes**: A packed type that contains 2 elements per byte and only supports conversion to and from DT_FP16.

3. **DT_FP4_E2M1/DT_FP4_E1M2 Special Notes**:
    <!-- npu="950" id6 -->
    - Ascend 950PR&950DT products: Supported
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - Atlas A3 products: Not Supported
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    - Atlas A2 products: Not Supported
    <!-- end id8 -->
    - Uses logical 4-bit types, consistent with scaled_mm; both the input parameter annotation and the cast target type use enums without the X2 suffix.
    - The logical shape remains unchanged before and after conversion; 2 FP4 elements are stored per byte, and the length of the logical last axis must be even.
    - When interacting with Torch, torch.uint8 carries the packed data, and the storage shape is [m, n/2]; after the parameter is annotated as `pypto.Tensor([], pypto.DT_FP4_E1M2)` or `pypto.Tensor([], pypto.DT_FP4_E2M1)`, the logical shape seen inside the kernel is [m, n].
    - Only supports conversion to and from DT_BF16.

4. **DT_HF8 (hifloat8) special notes**:
    <!-- npu="950" id9 -->
    - Ascend 950PR&950DT products: Supported
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - Atlas A3 products: Not Supported
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2 products: Not Supported
    <!-- end id11 -->
    - The CAST_ROUND rounding mode (corresponding to the hardware ROUND_A) must be used.
    - If another CastMode is specified, the implementation will automatically fall back to CAST_ROUND.

5. **Unsupported CastMode handling**:
    - When a user specifies a CastMode that is not supported by the hardware for a conversion, the framework does not report an error.
    - The framework automatically adopts the default CastMode for that conversion:
      - Floating-point to integer: CAST_TRUNC is used.
      - Other scenarios: CAST_RINT is used.

6. **deq mode description**: The INT32→FP16 conversion uses the deq mode, which requires setting the scaling factor through `set_deqscale`, with a default value of 1.0.
7. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2], pypto.DT_FP32)
y = pypto.cast(x, pypto.DT_FP16)
```

The results are as follows:

```python
Input data x: [2.0, 3.0] # x.dtype: pypto.DT_FP32

Output data y: [2.0, 3.0] # y.dtype: pypto.DT_FP16
```

#### Using the Saturation Mode (Recommended for Floating-Point to Integer Conversion)

```python
# Example 1: FP16 to INT8 conversion, using saturation mode to prevent overflow.
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
# Output: [127, -128, 50]

# Example 2: FP16 to INT8 conversion, with saturation mode disabled.
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.OFF)
# Output: [44, -44, 50]
```
