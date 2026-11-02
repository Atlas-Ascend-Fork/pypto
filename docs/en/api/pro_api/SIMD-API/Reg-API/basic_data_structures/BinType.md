# BinType

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:23:12.174Z pushedAt=2026-10-28T11:57:28.831Z nl=2 -->

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

BinType defines the index range for histogram statistics in [vf.histograms](../sorting_and_indexing/histograms.md), specifying which part of the source register's index range is counted for the element distribution.

## Prototype

```python
class BinType(enum.Enum):
     BIN0 = ...  # Lower-half index range [0-127] (default).
     BIN1 = ...  # Upper-half index range [128-255].
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, bin_type=pl.BinType.BIN0)
```
