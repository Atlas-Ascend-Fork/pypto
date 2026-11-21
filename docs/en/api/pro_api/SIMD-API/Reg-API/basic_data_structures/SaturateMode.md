# SaturateMode

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

SaturateMode defines the saturation handling mode used by [vf.astype](../type_conversion/astype.md) during narrowing type conversion. It controls the handling strategy when the source data exceeds the representable range of the destination type.

## Prototype

```python
class SaturateMode(enum.Enum):
     OFF = ...  # Non-saturation mode (default). Values beyond the range are truncated or the behavior varies depending on the conversion scenario.
     ON = ...  # Saturation mode. Values beyond the range are truncated to the maximum or minimum value of the destination type.
```

## Constraints

The behavior differences between the non-saturation mode and saturation mode in different type conversion scenarios are as follows:

| Scenario| Non-saturation Mode (OFF)| Saturation Mode (ON)|
|---|---|---|
| Floating-point to integer| When the input data exceeds the extreme value of the output type, the result is truncated to the data width of the target format (keeping the least significant bits). For example, an input half value of 257 produces a uint8_t value of 1. If the input is +/-inf, the corresponding extreme value of the output type is returned. If the input is nan, 0 is returned.| When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned. For example, an input half value of 257 produces a uint8 value of 255; an input half value of -inf produces a uint8_t value of 0. If the input is nan, 0 is returned.|
| Floating-point to floating-point| If the input is nan, the output is nan. If the input is +/-inf, the output is +/-inf.| If the input is nan, the output is 0. When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned.|
| Integer to floating-point| The non-saturation mode is not supported.| If the input is nan, the output is 0. When the input data exceeds the extreme value of the output type, the corresponding extreme value of the output type is returned. In this scenario, the saturation mode is used by default and does not need to be configured.|
| Integer to integer| The input data is truncated to the target data width. For example, an input int32_t value of 256 produces a uint8_t value of 0.| When the input data exceeds the range of the target data type, it is saturated to the extreme value of the target data type.|

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel(src, preg):
    dst = vf.astype(src, preg, dtype=pl.DT_INT8, saturate=pl.SaturateMode.ON)
```
