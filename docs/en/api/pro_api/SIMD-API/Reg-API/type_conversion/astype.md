# vf.astype

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

vf.astype is used for data type precision conversion. It converts the data type of the source operand to the data type of the destination operand, supporting float-to-integer, float-to-float, integer-to-float, and integer-to-integer data type conversions.

During the conversion, due to bit width and precision changes, the following parameters can be configured to implement the required functionality:

- **layout** (pypto_pro.language.CastLayout): When the bit widths of the source and destination operands are different, the amount of data processed by a single instruction is determined by the data type with the larger bit width. layout controls how the elements of the smaller-bit-width type are arranged in the register. pypto_pro.language.CastLayout.ZERO places elements at even index positions, and pypto_pro.language.CastLayout.ONE places elements at odd index positions. FP4 types also support the pypto_pro.language.CastLayout.TWO and pypto_pro.language.CastLayout.THREE modes (used for 4× expansion/narrowing).
- **saturate** (pypto_pro.language.SaturateMode): This parameter is used to set the saturation or non-saturation mode. In saturation mode, values outside the representable range of the target type are clamped to the maximum/minimum value of the target type; in non-saturation mode, the behavior for out-of-range values varies depending on the conversion scenario (see [Constraints](#constraints) for details).
- **mode** (pypto_pro.language.MergeMode): It specifies the mode for writing data into the register. Elements not selected by preg are either set to zero in dst (pypto_pro.language.MergeMode.ZEROING) or retain their original dst values (pypto_pro.language.MergeMode.MERGING). Currently, the device supports only the pypto_pro.language.MergeMode.ZEROING mode.
- **round_mode** (pypto_pro.language.VFRoundMode): It is used to set the rounding mode. This parameter takes effect only in conversions that may cause precision loss and that support this rounding mode.

Under different data types, the preg bit width corresponding to each element is different. During type conversion, mask_reg filters valid elements based on the input source operand. When the bit widths of the source and destination operands are different, the amount of data processed by a single instruction is determined by the data type with the larger bit width. layout controls how the elements of the smaller-bit-width type are arranged in the register.

The figure below shows the type conversion process between 16-bit-wide and 32-bit-wide types when mask_reg and layout act together.

**Figure 1** 16-bit-wide to 32-bit-wide type conversion process

![16-bit-wide to 32-bit-wide type conversion process](../../../figures/astype_b16_to_b32_conversion.svg)

**Figure 2** 32-bit-wide to 16-bit-wide type conversion process

![32-bit-wide to 16-bit-wide type conversion process](../../../figures/astype_b32_to_b16_conversion.svg)

In particular, for conversions between DT_FP4E2M1/DT_FP4E1M2 and DT_BF16, the instruction reads and writes elements in pairs of 2; when converting from a larger type to a smaller type, the valid bit of preg is determined by the even bit. The following figures show the conversion process between DT_FP4E2M1 and DT_BF16 when mask_reg and layout act together:

**Figure 3** Conversion from DT_FP4E2M1 to DT_BF16

![Conversion from DT_FP4E2M1 to DT_BF16](../../../figures/astype_fp4x2_e2m1_to_bf16_conversion.svg)

**Figure 4** Conversion from DT_BF16 to DT_FP4E2M1

![Conversion from DT_BF16 to DT_FP4E2M1](../../../figures/astype_bf16_to_fp4x2_e2m1_conversion.svg)

## Prototype

```python
astype(src, preg, /, *, dtype: DType, layout: Optional[CastLayout] = None, round_mode: Optional[VFRoundMode] = None, saturate: Optional[SaturateMode] = None, mode: Optional[MergeMode] = None) -> dst
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| src | Input| Source operand, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints).|
| preg | Input| [mask_reg](../basic_data_structures/mask_reg.md). preg filters valid elements based on the input source operand.|
| dtype | Input| (Mandatory) Data type of the destination register (for example, pypto_pro.language.DT_FP16 or pypto_pro.language.DT_INT32). Since the destination type is different from the source type after conversion, this parameter must be specified explicitly.|
| layout | Input| (Optional) [CastLayout](../basic_data_structures/CastLayout.md) enumeration type. pypto_pro.language.CastLayout.ZERO (even half, default) or pypto_pro.language.CastLayout.ONE (odd half). When the bit widths of the source and destination operands are different, this parameter controls how the elements of the smaller-bit-width type are arranged in the register. FP4 types also support the pypto_pro.language.CastLayout.TWO or pypto_pro.language.CastLayout.THREE mode. The supported values vary depending on the conversion path. For details, see the tables in [Constraints](#constraints).|
| round_mode | Input| (Optional) [VFRoundMode](../basic_data_structures/VFRoundMode.md) enumeration type, indicating the floating-point rounding mode. The default value is pypto_pro.language.VFRoundMode.CAST_RINT. Different conversion paths support different rounding modes. For details, see the tables in [Constraints](#constraints). For conversion paths that do not involve precision loss, round_mode is marked as UNKNOWN (and can be omitted).|
| saturate | Input| (Optional) [SaturateMode](../basic_data_structures/SaturateMode.md) enumeration type. pypto_pro.language.SaturateMode.OFF (default, non-saturation) or pypto_pro.language.SaturateMode.ON (saturation). The supported values vary depending on the conversion path. For details, see the tables in [Constraints](#constraints). Paths marked as UNKNOWN indicate that saturation/non-saturation selection is not involved.|
| mode | Input| (Optional) This parameter corresponds to the [MergeMode](../basic_data_structures/MergeMode.md) type.<br>- pypto_pro.language.MergeMode.ZEROING (default): Elements not selected by preg are set to 0 in dst.<br>- pypto_pro.language.MergeMode.MERGING: Not supported currently.|

## Constraints

- Data type constraints:

  **Table 1** Supported data type conversion

  | src | dst |
  |---|---|
  | DT_INT4 | DT_INT16, DT_FP16, and DT_BF16|
  | DT_INT8 | DT_INT16, DT_FP16, and DT_INT32|
  | DT_UINT8 | DT_UINT16, DT_FP16, and DT_UINT32|
  | DT_FP4E2M1 | DT_BF16 |
  | DT_FP4E1M2 | DT_BF16 |
  | DT_HF8 | DT_FP16 and DT_FP32|
  | DT_FP8E8M0 | DT_BF16 |
  | DT_FP8E5M2 | DT_FP32 |
  | DT_FP8E4M3FN | DT_FP32 |
  | DT_INT16 | DT_INT4, DT_UINT8, DT_FP16, DT_INT32, DT_UINT32, and DT_FP32|
  | DT_UINT16 | DT_UINT8 and DT_UINT32|
  | DT_FP16 | DT_INT4, DT_INT8, DT_UINT8, DT_HF8, DT_INT16, DT_BF16, DT_INT32, and DT_FP32|
  | DT_BF16 | DT_FP4E2M1, DT_FP4E1M2, DT_FP8E8M0, DT_FP16, DT_INT32, and DT_FP32|
  | DT_INT32 | DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, and DT_INT64|
  | DT_UINT32 | DT_UINT8, DT_INT16, and DT_UINT16|
  | DT_FP32 | DT_HF8, DT_FP8E5M2, DT_FP8E4M3FN, DT_INT16, DT_FP16, DT_BF16, DT_INT32, and DT_INT64|
  | DT_INT64 | DT_INT32 and DT_FP32|

- Parameter type conversion in different scenarios:

  **Table 2** Float-to-integer conversion

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_FP16 | DT_INT4 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_UINT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT16 | UNKNOWN | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_INT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT32 | UNKNOWN | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT64 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |

  **Table 3** Float-to-float conversion

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_HF8 | DT_FP16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_HF8 | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E4M3FN | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E5M2 | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E8M0 | DT_BF16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP4E2M1 | DT_BF16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP4E1M2 | DT_BF16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP16 | DT_HF8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_HYBRID |
  | DT_FP16 | DT_BF16 | UNKNOWN | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP16 | UNKNOWN | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP4E2M1 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP4E1M2 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP8E8M0 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_BF16 | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_FP32 | DT_HF8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_HYBRID |
  | DT_FP32 | DT_FP8E4M3FN | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT |
  | DT_FP32 | DT_FP8E5M2 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT |
  | DT_FP32 | DT_FP16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_ODD/pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_BF16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |

  **Table 4** Integer-to-float conversion

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_INT4 | DT_FP16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT4 | DT_BF16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_FP16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_FP16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_FP16 | UNKNOWN | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_INT16 | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_FP32 | UNKNOWN | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |
  | DT_INT64 | DT_FP32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | pypto_pro.language.VFRoundMode.CAST_RINT/pypto_pro.language.VFRoundMode.CAST_ROUND/pypto_pro.language.VFRoundMode.CAST_FLOOR/pypto_pro.language.VFRoundMode.CAST_CEIL/pypto_pro.language.VFRoundMode.CAST_TRUNC |

  **Table 5** Integer-to-integer conversion

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_INT4 | DT_INT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_INT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_INT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_UINT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_UINT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_INT4 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_UINT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_INT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_UINT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT16 | DT_UINT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT16 | DT_UINT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_UINT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_INT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_UINT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_INT64 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | UNKNOWN | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_UINT8 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE/pypto_pro.language.CastLayout.TWO/pypto_pro.language.CastLayout.THREE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_INT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_UINT16 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |
  | DT_INT64 | DT_INT32 | pypto_pro.language.CastLayout.ZERO/pypto_pro.language.CastLayout.ONE | pypto_pro.language.SaturateMode.OFF/pypto_pro.language.SaturateMode.ON | pypto_pro.language.MergeMode.ZEROING | UNKNOWN |

  > **Meaning of UNKNOWN in the table**: If layout is UNKNOWN, the source and destination bit widths of the conversion path are the same, and the layout does not need to be specified (it can be omitted). If saturate is UNKNOWN, the conversion path does not involve saturation/non-saturation selection (it can be omitted). If round_mode is UNKNOWN, the conversion path does not involve precision loss (it can be omitted).

- Description of the saturation and non-saturation modes:

  **Table 6** Non-saturation mode and saturation mode in different type conversion scenarios

  | Scenario| Non-saturation Mode| Saturation Mode|
  |---|---|---|
  | Float-to-integer conversion| When the input data exceeds the extreme value of the output type, the result is truncated to the data width of the target format (keeping the least significant bits). For example, an input half value of 257 produces a uint8_t value of 1. If the input is +/-inf, the corresponding extreme value of the output type is returned. If the input is nan, 0 is returned.| When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned. For example, an input half value of 257 produces a uint8 value of 255; an input half value of -inf produces a uint8_t value of 0. If the input is nan, 0 is returned.|
  | Float-to-float conversion| If the input is nan, the output is nan. If the input is +/-inf, the output is +/-inf.| If the input is nan, the output is 0. When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned.|
  | Integer-to-float conversion| The non-saturation mode is not supported.| If the input is nan, the output is 0. When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned. In this scenario, the saturation mode is used by default and does not need to be configured.|
  | Integer-to-integer conversion| The input data is truncated to the target data width. For example, an input int32_t value of 256 produces a uint8_t value of 0.| When the input data exceeds the range of the target data type, it is saturated to the extreme value of the target data type.|

- Special constraints for float-to-float conversion:

    - When the output type is FP32, only the non-saturation mode is supported.
    - Non-saturation mode: When the output type is FP8E4M3FN, the output is NaN because FP8E4M3FN does not have an inf representation format.
    - Saturation mode: When the output type is FP8E5M2/FP8E4M3FN, an input NaN produces 0 by default. If CTRL[50] is set to 1'b1, the output is NaN.
    - The FP4E2M1/FP4E1M2 data type has no definitions for inf or NaN. For BF16-to-FP4 conversion, if an input BF16 value is inf or exceeds the maximum value range of FP4, the FP4 maximum value with the corresponding sign is returned; if the input is NaN, the FP4 output is 0
    - For the FP8E8M0 type, if the input BF16 value is +/-inf or its absolute value exceeds the maximum value of the FP8E8M0 type, the FP8E8M0 maximum value 0b11111110 is returned; if the input BF16 value is NaN, the FP8E8M0 NaN = 0b11111111 is output.

- Special constraints for integer-to-integer conversion:

    - For conversion from a narrow data type (such as INT16 (2 bytes)) to a wide data type (such as UINT32 (4 bytes)), only the saturation mode is supported. A negative input value will be saturated to 0.

## Return Value

Returns the destination operand dst, which is of type [reg_tensor](../basic_data_structures/reg_tensor.md). For details about the supported data types, see [Constraints](#constraints). When the bit width of the destination operand is smaller than that of the source operand, the invalid elements in the destination operand are all 0s under the influence of preg and layout.

## Examples

### BF16 to FP32

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_bf16 = vf.astype(reg_a, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    reg_f32 = vf.astype(reg_bf16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.bfloat16).to(torch.float32), rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### FP32 to FP16

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp32_to_fp16(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_f16 = vf.astype(reg_a, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
    reg_f32 = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel_fp32_to_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp32_to_fp16(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp32_to_fp16():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp32_to_fp16[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.float16).to(torch.float32), rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_fp32_to_fp16()
    print("PASSED")
```

### FP32 to INT32

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_round(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_i = vf.astype(reg_a, preg, dtype=pl.DT_INT32, round_mode=pl.VFRoundMode.CAST_RINT)
    reg_f = vf.astype(reg_i, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f, preg)

@pl.jit()
def example_kernel_round(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_round(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_round[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.int32).to(torch.float32), rtol=0, atol=1.0)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### FP32 to FP8E4M3FN

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_f8 = vf.astype(reg_a, preg, dtype=pl.DT_FP8E4M3FN, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.ON)
    reg_f32 = vf.astype(reg_f8, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)

@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp8(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_fp8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.to(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(out[:, ::2], expected[:, ::2], rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```

### BF16 to FP4E2M1

```python
import numpy as np
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp4(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    reg_a = vf.load_align(src_tile, 0, dtype=pl.DT_BF16)
    reg_f4 = vf.astype(reg_a, preg, dtype=pl.DT_FP4E2M1, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
    reg_bf16 = vf.astype(reg_f4, preg, dtype=pl.DT_BF16)
    vf.store_align(dst_tile, reg_bf16, preg)

@pl.jit()
def example_kernel_fp4(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_fp4(in_a, t_out)
        pl.store(out, t_out, [0, 0])

_FP4_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_E2M1_EVEN_MASK = np.array([True, False, True, False, True, False, True, False])

def test_example_fp4():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.bfloat16)
    out = torch.empty([1, 128], device=device, dtype=torch.bfloat16)
    example_kernel_fp4[None, core_nums](a, out)
    torch.npu.synchronize()
    a_np = a.cpu().float().numpy()
    sign = np.sign(a_np)
    abs_val = np.abs(a_np)
    dist = np.abs(abs_val[..., None] - _FP4_E2M1_VALUES)
    min_dist = dist.min(axis=-1)
    is_nearest = dist == min_dist[..., None]
    candidates = is_nearest & _FP4_E2M1_EVEN_MASK
    has_even = candidates.any(axis=-1)
    idx = np.where(has_even, np.argmax(candidates, axis=-1), np.argmin(dist, axis=-1))
    expected = torch.from_numpy(sign * _FP4_E2M1_VALUES[idx]).to(device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(out[:, ::2], expected[:, ::2], rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    test_example_fp4()
    print("PASSED")
```

### FP16 to HF8

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_a = vf.load_align(src_tile, 0, dtype=pl.DT_FP16)
    reg_hf8 = vf.astype(reg_a, preg, dtype=pl.DT_HF8, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_ROUND, saturate=pl.SaturateMode.ON)
    reg_f16 = vf.astype(reg_hf8, preg, dtype=pl.DT_FP16)
    vf.store_align(dst_tile, reg_f16, preg)

@pl.jit()
def example_kernel_hf8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_hf8(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_hf8[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out[:, ::2], a[:, ::2], rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```

### INT64 Data Type Example

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_int64(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT64)
    reg_a = vf.load_align(src_tile, 0)
    reg_f32 = vf.astype(reg_a, preg, dtype=pl.DT_FP32)
    reg_out = vf.astype(reg_f32, preg, dtype=pl.DT_INT64)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_int64(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT64],
):
    tf = pl.TileType(shape=[1, 32], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=256, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        example_vf_int64(in_a, t_out)
        pl.store(out, t_out, [0, 0])

def test_example_int64():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randint(-100, 100, [1, 32], device=device, dtype=torch.int64)
    out = torch.empty([1, 32], device=device, dtype=torch.int64)
    example_kernel_int64[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_int64()
    print("PASSED")
```
