# BinType

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

BinType defines the index range for histogram statistics in [vf.histograms](../sorting_and_indexing/histograms.md). It is used to specify which part of the index range in the source register is counted for the element distribution.

## Prototype

```python
class BinType(enum.Enum):
     BIN0 = ...  # Lower-half index range [0–127] (default).
     BIN1 = ...  # Upper-half index range [128–255].
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, bin_type=pl.BinType.BIN0)
```
