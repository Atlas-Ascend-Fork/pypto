# MergeMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:28:26.534Z pushedAt=2026-10-28T11:57:28.877Z nl=2 -->

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

MergeMode defines how mask-unselected elements (inactive elements) in a VF compute instruction are handled in the destination register. It applies to VF compute APIs such as [vf.add](../basic_arithmetic/add.md), [vf.sub](../basic_arithmetic/sub.md), [vf.mul](../basic_arithmetic/mul.md), and [vf.div](../basic_arithmetic/div.md).

## Prototype

```python
class MergeMode(enum.Enum):
     ZEROING = ...  # Zero out positions not selected by mask (default).
     MERGING = ...  # Keep the original value of the destination register at positions not selected by mask.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.add(src0, src1, preg, mode=pl.MergeMode.ZEROING)
```
