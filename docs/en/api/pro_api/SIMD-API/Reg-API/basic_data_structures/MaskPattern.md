# MaskPattern

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:26:48.764Z pushedAt=2026-10-28T11:57:28.860Z nl=2 -->

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

MaskPattern defines the mask generation mode for [vf.create_mask](../mask_operations/create_mask.md) and [vf.mask_gen_with_reg_tensor](../mask_operations/mask_gen_with_reg_tensor.md), which controls which elements in the register are marked as valid.

## Prototype

```python
class MaskPattern(enum.Enum):
     ALL = ...  # All elements are valid.
     ALLF = ...  # All elements are invalid.
     VL1 = ...  # The lowest 1 element is valid.
     VL2 = ...  # The lowest 2 elements are valid.
     VL3 = ...  # The lowest 3 elements are valid.
     VL4 = ...  # The lowest 4 elements are valid.
     VL8 = ...  # The lowest 8 elements are valid.
     VL16 = ...  # The lowest 16 elements are valid.
     VL32 = ...  # The lowest 32 elements are valid.
     VL64 = ...  # The lowest 64 elements are valid.
     VL128 = ...  # The lowest 128 elements are valid.
     M3 = ...  # The first of every 3 elements is valid.
     M4 = ...  # The first of every 4 elements is valid.
     H = ...  # The lower half is valid.
     Q = ...  # The lower quarter is valid.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.VL8)  # The lowest 8 elements are valid.
```
