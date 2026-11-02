# StoreDist

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:33:24.960Z pushedAt=2026-10-28T11:57:28.897Z nl=2 -->

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

StoreDist defines the data storage distribution mode of [vf.store_align](../data_movement/store_align.md), which controls how data is moved from registers to the UB.

## Prototype

```python
class StoreDist(enum.Enum):
     # reg_tensor single-load mode
     NORM = ...  # Normal aligned storage (default). The bit-width granularity is automatically selected based on dtype. The 64-bit data types DT_INT64/DT_UINT64 support only this mode.
     NORM_B8 = ...  # Normal storage by 8-bit-wide type
     NORM_B16 = ...  # Normal storage by 16-bit-wide type
     NORM_B32 = ...  # Normal storage by 32-bit-wide type
     FIRST_ELEMENT = ...  # Store only lane 0 (the first element), with the bit width granularity automatically selected based on dtype.
     FIRST_ELEMENT_B8 = ...  # Store only the first element as an 8-bit-wide type.
     FIRST_ELEMENT_B16 = ...  # Store only the first element as a 16-bit-wide type.
     FIRST_ELEMENT_B32 = ...  # Store only the first element as a 32-bit-wide type.
     PACK = ...  # Packed storage. Store the lower half of the bit data of the valid elements in src contiguously in dst based on mask, with the bit width granularity automatically selected based on dtype.
     PACK_B16 = ...  # Packed storage as a 16-bit-wide type.
     PACK_B32 = ...  # Packed storage as a 32-bit-wide type.
     PACK_B64 = ...  # Packed storage at B64 granularity.
     PACK4 = ...  # 4-element packed storage. Store the lower 8 bits of the valid elements contiguously (32-bit-wide type).
     PACK4_B32 = ...  # 4-element packed storage as a 32-bit-wide type.
     # reg_tensor dual load mode
     INTLV = ...  # Interleaved storage, which interleaves the elements in src0 and src1 (the bit width type is automatically selected based on dtype).
     INTLV_B8 = ...  # Interleaved storage by 8-bit-wide type.
     INTLV_B16 = ...  # Interleaved storage by 16-bit-wide type.
     INTLV_B32 = ...  # Interleaved storage by 32-bit-wide type.
     # mask_reg mode
     # NORM is the same as above, moving out VL/8 data.
     # PACK is the same as above, discarding data at 1-bit intervals and compressing VL/8 data into VL/16 for load.
```

## Constraints

The supported modes of StoreDist vary depending on the target mode:

- **reg_tensor single load mode**: Supports NORM (including NORM_B8/NORM_B16/NORM_B32), FIRST_ELEMENT (including FIRST_ELEMENT_B8/B16/B32), PACK (including PACK_B16/B32/B64), and PACK4 (including PACK4_B32).
- **reg_tensor dual load mode**: Supports INTLV (including INTLV_B8/INTLV_B16/INTLV_B32) (requires two source registers)
- **mask_reg mode**: Supports NORM and PACK (explicit granularity forms are not supported)

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, dist=pl.StoreDist.NORM)
```
