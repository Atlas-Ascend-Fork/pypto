# VFRoundMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:33:40.547Z pushedAt=2026-10-28T11:57:28.900Z nl=2 -->

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

VFRoundMode defines the floating-point rounding mode used in [vf.astype](../type_conversion/astype.md) type conversion. It takes effect only in conversions that may cause precision loss and support the rounding mode. Conversion paths that do not involve precision loss are marked as UNKNOWN (which can be omitted).

Different conversion paths support different rounding modes. For details, see the tables in [vf.astype constraints](../type_conversion/astype.md#constraints).

## Prototype

```python
class VFRoundMode(enum.Enum):
     CAST_ROUND = ...  # Round to the nearest integer, with ties rounding away from zero.
     CAST_RINT = ...  # Default rounding, rounding to the nearest even number.
     CAST_FLOOR = ...  # Round down (floor).
     CAST_CEIL = ...  # Round up (ceil).
     CAST_TRUNC = ...  # Truncate toward zero.
     CAST_ODD = ...  # Round to odd (Von Neumann rounding).
     CAST_HYBRID = ...  # Hybrid rounding (supported only on Ascend 950PR/DT).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT32, round_mode=pl.VFRoundMode.CAST_FLOOR)
```
