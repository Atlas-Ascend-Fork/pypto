# SaturateMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:31:58.270Z pushedAt=2026-10-28T11:57:28.894Z nl=2 -->

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

SaturateMode defines the saturation handling mode for narrowing type conversion in [vf.astype](../type_conversion/astype.md), used to control the processing strategy when the source data exceeds the representable range of the target type.

## Prototype

```python
class SaturateMode(enum.Enum):
     OFF = ...  # Non-saturation mode (default). Values exceeding the range are truncated, or the behavior varies by conversion scenario.
     ON = ...  # Saturation mode. Values exceeding the range are truncated to the maximum or minimum value of the target type.
```

## Constraints

The behavior differences between the non-saturation mode and the saturation mode in different type conversion scenarios are as follows:

| Scenario | Non-saturation mode (OFF) | Saturation mode (ON) |
|---|---|---|
| Float to integer | When the input data exceeds the output type limits, the result is truncated to the data width of the target format (retaining the least significant bits). For example, if the input half value is 257, the output uint8_t value is 1. If the input is +/-inf, the corresponding limits of the output type are returned. If the input is nan, 0 is returned. | When the input data exceeds the output type limits, the corresponding limits of the output type are returned. For example, if the input half value is 257, the output uint8 value is 255. If the input half value is -inf, the output uint8_t value is 0. If the input is nan, 0 is returned. |
| Float to float | If the input data is nan, the output is nan. If the input is +/-inf, the output is +/-inf. | If the input is nan, the output is 0. When the input data exceeds the output type limits, the corresponding limits of the output type are returned. |
| Integer to float | Non-saturation mode is not supported. | If the input is nan, the output is 0. When the input data exceeds the output type limits, the corresponding limits of the output type are returned. This scenario defaults to saturation mode and requires no configuration. |
| Integer to integer | The input data is truncated to the target data width. For example, if the input int32_t value is 256, the output uint8_t value is 0. | If the input data exceeds the target data range, it is saturated to the target type limits. |

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT8, saturate=pl.SaturateMode.ON)
```
