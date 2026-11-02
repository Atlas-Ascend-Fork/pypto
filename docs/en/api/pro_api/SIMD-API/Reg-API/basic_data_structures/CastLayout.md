# CastLayout

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:23:42.980Z pushedAt=2026-10-28T11:57:28.846Z nl=2 -->

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

CastLayout defines how elements with a smaller bit width are arranged in registers when the source and destination operands in [vf.astype](../type_conversion/astype.md), [vf.exp_sub](../composite_computation/exp_sub.md), and [vf.muls_cast](../composite_computation/muls_cast.md) have different bit widths.

The computation amount of a single instruction is determined by the data type with the larger bit width. layout controls the arrangement position of elements with a smaller bit width in registers.

## Prototype

```python
class CastLayout(enum.Enum):
     ZERO = ...  # Write the result to the even half region (PART_EVEN).
     ONE = ...  # Write the result to the odd half region (PART_ODD).
     TWO = ...  # Third half region (used for 4x expansion/narrowing of the FP4 type).
     THREE = ...  # Fourth half region (used for 4x expansion/narrowing of the FP4 type).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT8, layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
```
