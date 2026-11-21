# DuplicatePos

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

DuplicatePos defines the element duplication position selection in the broadcast mode of [vf.full](../data_movement/full.md). It is used to specify which indexed element is broadcast to the entire register.

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
