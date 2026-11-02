# SaturationMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:50:04.122Z pushedAt=2026-10-28T11:57:29.686Z nl=2 -->

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

**SaturationMode** defines the overflow handling method for floating-point to integer conversion. It controls the processing strategy when the source data exceeds the representable range of the target integer type, ensuring the correctness and predictability of the conversion result.

## Prototype

```python
class SaturationMode(enum.Enum):
     OFF = ...   # Truncation mode (default). Directly truncates the excess part, which may cause overflow.
     ON = ...    # Saturation mode. Values out of range are truncated to the maximum or minimum value of the target type.
```
