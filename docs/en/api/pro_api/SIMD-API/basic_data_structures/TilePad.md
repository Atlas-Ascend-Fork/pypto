# pypto_pro.language.TilePad

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

Enumerates the padding modes when the tile boundary is insufficient. It is used in the tail block or non-full block scenario.

When the valid data region of a tile is smaller than its shape (for example, the tail block of a dynamic dimension), the portion that exceeds the valid region needs to be padded in the specified mode.

## Prototype

```python
PYPTO_DECLARE_ENUM(
    TilePad,
    null,
    zero,
    max,
    min
)
```

## Parameters

| Value| Description|
|---|---|
| null | No padding. This is the default value and is applicable to scenarios where the invalid region does not need to be processed.|
| zero | Padding with zeros. It is typically used for convolutional padding and zero initialization.|
| max | Padding with the maximum value of the corresponding data type. It is typically used for the invalid regions in the minimum value operations.|
| min | Padding with the minimum value of the corresponding data type. It is typically used for the invalid regions of the maximum value operations.|

## Constraints

None

## Examples

### Tile Padding Modes

```python
import pypto_pro.language as pl
# No padding (default)
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec)

# Padding with zeros
tt_pad = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
```
