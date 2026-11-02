# LoadDist

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:25:51.973Z pushedAt=2026-10-28T11:57:28.863Z nl=2 -->

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

LoadDist defines the data load distribution mode of [vf.load_align](../data_movement/load_align.md), which controls how data is moved from UB to registers. Different modes correspond to different hardware instructions, affecting the data layout in registers and the broadcast/sampling behavior.

## Prototype

```python
class LoadDist(enum.Enum):
     # RegTensor target
     NORM = ...  # Normal element-wise load (default).
     BRC = ...  # Broadcast a single element to the entire register.
     BRC_B8 = ...  # Broadcast by 8-bit-wide type.
     BRC_B16 = ...  # Broadcast by 16-bit-wide type.
     BRC_B32 = ...  # Broadcast by 32-bit-wide type.
     US = ...  # Upsample (repeat each element twice).
     US_B8 = ...  # Upsample by 8-bit-wide type.
     US_B16 = ...  # Upsample by 16-bit-wide type.
     DS = ...  # Downsample (discard every other element).
     DS_B8 = ...  # Downsample by 8-bit-wide type.
     DS_B16 = ...  # Downsample by 16-bit-wide type.
     UNPK = ...  # Unpack.
     UNPK_B8 = ...  # Unpack by 8-bit-wide type.
     UNPK_B16 = ...  # Unpack by 16-bit-wide type.
     UNPK_B32 = ...  # Unpack by 32-bit-wide type.
     UNPK4 = ...  # Unpack 4 elements.
     BLK = ...  # Block copy.
     E2B = ...  # Extend from 16-bit-wide type to 32-bit-wide type.
     E2B_B16 = ...  # Extend by 16-bit-wide type.
     E2B_B32 = ...  # Extend at 32-bit-wide type granularity.
     DINTLV_B8 = ...  # Deinterleave by 8-bit-wide type (split odd/even registers).
     DINTLV_B16 = ...  # Deinterleave by 16-bit-wide type.
     DINTLV_B32 = ...  # Deinterleave by 32-bit-wide type.
     # MaskReg target.
     # NORM/US/DS are the same as the RegTensor target, with a data transfer amount of VL/8 bytes.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    reg = vf.load_align(ub_tile, dist=pl.LoadDist.BRC)
```
