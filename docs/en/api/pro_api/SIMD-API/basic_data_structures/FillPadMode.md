# pypto_pro.language.FillPadMode

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

Enumerates the padding modes used by the fillpad API. It is used to specify the address and shape relationship between the source tile and the destination tile.

## Prototype

```python
class FillPadMode(enum.Enum):
    NORMAL = 0   # The source tile and destination tile have the same shape but different addresses.
    EXPAND = 1   # The shape of each dimension of the destination tile is not smaller than that of the source tile, and the extended region is padded.
    INPLACE = 2  # The source tile and destination tile have the same shape and share the same address.
```
