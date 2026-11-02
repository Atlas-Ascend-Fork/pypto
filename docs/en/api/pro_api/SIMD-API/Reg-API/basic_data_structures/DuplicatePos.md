# DuplicatePos

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:24:18.266Z pushedAt=2026-10-28T11:57:28.842Z nl=2 -->

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

DuplicatePos defines the element copy position selection in the broadcast mode of [vf.full](../data_movement/full.md), which is used to specify the element at which index position is broadcast to the entire register.

## Prototype

```python
class DuplicatePos(enum.Enum):
     LOWEST = ...  # Broadcast the element at the lowest index position (default).
     HIGHEST = ...  # Broadcast the element at the highest index position.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.full(preg, pos=pl.DuplicatePos.LOWEST)
```
