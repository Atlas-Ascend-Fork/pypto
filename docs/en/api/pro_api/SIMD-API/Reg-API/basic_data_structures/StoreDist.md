# StoreDist

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

StoreDist defines the data storage distribution mode for [vf.store_align](../data_movement/store_align.md). It controls how data is moved from registers to UB.

## Prototype

```python
class StoreDist(enum.Enum):
     # Single-store-out mode for reg_tensor
     NORM = ...  # Normal aligned store (default); the bit-width granularity is selected automatically based on dtype. 64-bit data types DT_INT64 and DT_UINT64 support only this mode.
     NORM_B8 = ...  # Normal store with 8-bit element width.
     NORM_B16 = ...  # Normal store with 16-bit element width.
     NORM_B32 = ...  # Normal store with 32-bit element width.
     FIRST_ELEMENT = ...  # Store only lane 0 (the first element); the bit-width granularity is selected automatically based on dtype.
     FIRST_ELEMENT_B8 = ...  # Store only the first element with 8-bit element width.
     FIRST_ELEMENT_B16 = ...  # Store only the first element with 16-bit element width.
     FIRST_ELEMENT_B32 = ...  # Store only the first element with 32-bit element width.
     PACK = ...  # Packed store: according to the mask, the least significant half of bits of valid elements in src are stored contiguously in dst; the bit-width granularity is selected automatically based on dtype.
     PACK_B16 = ...  # Packed store with 16-bit element width.
     PACK_B32 = ...  # Packed store with 32-bit element width.
     PACK_B64 = ...  # Packed store with B64 granularity.
     PACK4 = ...  # 4-element packed store: the lower 8 bits of valid elements are stored contiguously (32-bit element width).
     PACK4_B32 = ...  # 4-element packed store with 32-bit element width.
     # Dual-store-out mode for reg_tensor
     INTLV = ...  # Interleaved store: Elements from src0 and src1 are stored in an interleaved manner (the bit-width type is selected automatically based on dtype).
     INTLV_B8 = ...  # Interleaved store with 8-bit element width.
     INTLV_B16 = ...  # Interleaved store with 16-bit element width.
     INTLV_B32 = ...  # Interleaved store with 32-bit element width.
     # mask_reg mode
     # NORM is the same as above; the moved data volume is VL/8.
     # PACK is the same as above; every other bit is discarded, and VL/8 of data is compressed to VL/16 before being moved out.
```

## Constraints

StoreDist supports different modes depending on the target mode:

- **reg_tensor single-store-out mode**: supports NORM (including NORM_B8, NORM_B16, and NORM_B32), FIRST_ELEMENT (including FIRST_ELEMENT_B8, FIRST_ELEMENT_B16, and FIRST_ELEMENT_B32), PACK (including PACK_B16, PACK_B32, and PACK_B64), and PACK4 (including PACK4_B32).
- **reg_tensor dual-store-out mode**: supports INTLV (including INTLV_B8, INTLV_B16, and INTLV_B32), and requires two source registers.
- **mask_reg mode**: supports NORM and PACK (explicit granularity is not supported).

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, dist=pl.StoreDist.NORM)
```
