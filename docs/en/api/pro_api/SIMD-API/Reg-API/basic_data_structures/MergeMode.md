# MergeMode

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

MergeMode defines how to process the elements (inactive elements) that are not selected by mask in the destination register in the VF computation instruction. It is applicable to VF computation APIs such as [vf.add](../basic_arithmetic/add.md), [vf.sub](../basic_arithmetic/sub.md), [vf.mul](../basic_arithmetic/mul.md), and [vf.div](../basic_arithmetic/div.md).

## Prototype

```python
class MergeMode(enum.Enum):
     ZEROING = ...  # Set the positions that are not selected by mask to 0 (default).
     MERGING = ...  # Retain the original value of the target register for the positions that are not selected by mask.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.add(src0, src1, preg, mode=pl.MergeMode.ZEROING)
```
