# PackPart

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:28:38.079Z pushedAt=2026-10-28T11:57:28.881Z nl=2 -->

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

PackPart defines the half-region selection for vf.pack and vf.unpack, specifying whether the pack or unpack operation processes the upper half or the lower half of a register.

## Prototype

```python
class PackPart(enum.Enum):
     LOWER = ...  # Lower half (default)
     UPPER = ...  # Upper half (only RegTraitNumTwo supports LOWER)
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.pack(src0, src1, preg, part=pl.PackPart.LOWER)
```
