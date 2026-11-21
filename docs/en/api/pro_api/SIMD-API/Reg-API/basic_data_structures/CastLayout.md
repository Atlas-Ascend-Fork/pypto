# CastLayout

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

CastLayout defines the layout of elements with a smaller bit width in a register when the source and destination operands of [vf.astype](../type_conversion/astype.md), [vf.exp_sub](../composite_computation/exp_sub.md), and [vf.muls_cast](../composite_computation/muls_cast.md) have different bit widths.

The execution width of a single instruction is determined by the data type with a larger bit width. layout controls how the elements with a smaller bit width are arranged in registers.

## Prototype

```python
class CastLayout(enum.Enum):
     ZERO = ...  # Write the result to the even half (PART_EVEN).
     ONE = ...  # Write the result to the odd half (PART_ODD).
     TWO = ...  # Third half (used for FP4 4x expansion/narrowing).
     THREE = ...  # Fourth half (used for FP4 4x expansion/narrowing).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel(src, preg):
    dst = vf.astype(src, preg, dtype=pl.DT_INT8, layout=pl.CastLayout.ZERO,
                    round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.OFF)
```
