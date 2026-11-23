# VFRoundMode

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

VFRoundMode defines the floating-point rounding mode used by [vf.astype](../type_conversion/astype.md) during type conversion. It takes effect only for conversions that may cause precision loss and support the specified rounding mode. Conversion paths that do not involve precision loss are marked as UNKNOWN (and can be omitted).

Different conversion paths support different rounding modes. For details, refer to [vf.astype Constraints](../type_conversion/astype.md#constraints).

## Prototype

```python
class VFRoundMode(enum.Enum):
     CAST_ROUND = ...  # Round half away from zero.
     CAST_RINT = ...  # Default rounding: round to nearest, ties to even.
     CAST_FLOOR = ...  # Round toward negative infinity (floor).
     CAST_CEIL = ...  # Round toward positive infinity (ceil).
     CAST_TRUNC = ...  # Round toward zero (truncation).
     CAST_ODD = ...  # von Neumann rounding: round to nearest, ties to odd.
     CAST_HYBRID = ...  # Hybrid rounding.
```

## Constraints

None

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel(src, preg):
    dst = vf.astype(src, preg, dtype=pl.DT_INT32, round_mode=pl.VFRoundMode.CAST_FLOOR)
```
