# pypto_pro.language.TilePad

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:49:37.844Z pushedAt=2026-10-28T11:57:28.646Z nl=2 -->

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

Enumeration of padding modes used when the Tile boundary is insufficient, for tail blocks or non-full blocks.

When the valid data region of a Tile is smaller than its shape (for example, a tail block of a dynamic dimension), the portion beyond the valid region is padded according to the specified mode.

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

| parameter value | Description |
|---|---|
| null | No padding. This is the default value and applies to scenarios where invalid regions do not need to be processed. |
| zero | Pads with zeros. Typically used for convolution padding and zero initialization. |
| max | Pads with the maximum value of the corresponding data type. Typically used for invalid regions in minimum-value operations. |
| min | Pads with the minimum value of the corresponding data type. Typically used for invalid regions in maximum-value operations. |

## Constraints

None

## Examples

### Tile Padding Mode

```python
import pypto_pro.language as pl
# No padding (default).
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec)

# Zero padding.
tt_pad = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
```
