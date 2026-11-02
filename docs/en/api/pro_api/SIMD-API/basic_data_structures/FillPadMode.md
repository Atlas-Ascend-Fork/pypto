# pypto_pro.language.FillPadMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:45:39.934Z pushedAt=2026-10-28T11:57:28.620Z nl=2 -->

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

Fill mode enumeration used by the fillpad API to specify the address and shape relationship between the source tile and the destination tile.

## Prototype

```python
class FillPadMode(enum.Enum):
    NORMAL = 0   # The source tile and destination tile have the same shape but different addresses.
    EXPAND = 1   # The destination tile has a shape no smaller than the source tile in each dimension, and the expanded area is filled.
    INPLACE = 2  # The source tile and destination tile have the same shape and share the same address.
```
