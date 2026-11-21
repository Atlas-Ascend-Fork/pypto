# PackPart

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

PackPart defines the half selection for vf.pack and vf.unpack, specifying whether the pack or unpack operation processes the upper half or the lower half of the register.

## Prototype

```python
class PackPart(enum.Enum):
     LOWER = ...  # Lower half (default)
     UPPER = ...  # Upper half (Only RegTraitNumTwo supports LOWER.)
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel(src):
    dst = vf.pack(src, part=pl.PackPart.LOWER, dtype=pl.DT_UINT8)
```
