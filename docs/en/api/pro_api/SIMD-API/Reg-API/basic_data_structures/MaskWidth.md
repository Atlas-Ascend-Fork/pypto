# MaskWidth

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:27:25.519Z pushedAt=2026-10-28T11:57:28.867Z nl=2 -->

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

MaskWidth defines the bit-width expansion mode used by [vf.get_mask_spr](../data_movement/get_mask_spr.md) when reading the SPR mask, controlling the bit expansion granularity when reading mask data from the MASK register.

## Prototype

```python
class MaskWidth(enum.Enum):
     B32 = ...  # Read the 64-bit MASK0, expanding each bit to 4 bits (movp_b32, default).
     B16 = ...  # Read the 128-bit {MASK1,MASK0}, expanding each bit to 2 bits (movp_b16).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.get_mask_spr(width=pl.MaskWidth.B32)
```
