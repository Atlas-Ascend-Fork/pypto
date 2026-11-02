# SqueezeMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:31:08.675Z pushedAt=2026-10-28T11:57:28.890Z nl=2 -->

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

SqueezeMode defines the data collection mode of [vf.squeeze](../comparison_and_selection/squeeze.md), which controls whether to store the byte count of valid elements in the AR SPR (Special Purpose Register).

## Prototype

```python
class SqueezeMode(enum.Enum):
     NO_STORE_REG = ...  # Do not store the byte count of valid elements in the AR SPR (default).
     STORE_REG = ...  # Store the total byte count of valid elements in the AR SPR.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.squeeze(src, preg, mode=pl.SqueezeMode.STORE_REG)
```
