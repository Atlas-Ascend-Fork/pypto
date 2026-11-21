# SaturationMode

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

SaturationMode defines the overflow handling method when floating-point numbers are converted to integers. It is used to control the processing strategy when the source data exceeds the representable range of the target integer type, ensuring the correctness and predictability of the conversion results.

## Prototype

```python
class SaturationMode(enum.Enum):
     OFF = ...   # Truncation mode (default). The excess part is directly truncated, which may cause overflow.
     ON = ...    # Saturation mode. The value that exceeds the range is truncated to the maximum or minimum value of the target type.
```
