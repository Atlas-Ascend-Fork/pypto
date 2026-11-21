# MaskWidth

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

MaskWidth defines the bit-width expansion mode when [vf.get_mask_spr](../data_movement/get_mask_spr.md) reads the SPR mask. It is used to control the bit expansion granularity when mask data is read from the MASK register.

## Prototype

```python
class MaskWidth(enum.Enum):
     B32 = ...  # Read 64-bit MASK0, with each bit expanded to 4 bits (movp_b32, default).
     B16 = ...  # Read 128-bit {MASK1, MASK0}, with each bit expanded to 2 bits (movp_b16).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.get_mask_spr(width=pl.MaskWidth.B32)
```
