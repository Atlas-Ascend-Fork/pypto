# SqueezeMode

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

SqueezeMode defines the data collection mode for [vf.squeeze](../comparison_and_selection/squeeze.md). It controls whether the byte count of valid elements is stored in the AR Special Purpose Register (SPR).

## Prototype

```python
class SqueezeMode(enum.Enum):
     NO_STORE_REG = ...  # Do not store the byte count of valid elements in the AR SPR (default).
     STORE_REG = ...  # Store the total number of bytes of valid elements in the AR SPR.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel(src, preg):
    dst = vf.squeeze(src, preg, gather_mode=pl.SqueezeMode.STORE_REG)
```
