# HistType

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

HistType defines the histogram statistics mode for [vf.histograms](../sorting_and_indexing/histograms.md). It specifies whether cumulative statistics or frequency statistics are used.

## Prototype

```python
class HistType(enum.Enum):
     ACCUMULATE = ...  # Cumulative statistics (default). The statistics are added to the existing values in the destination register.
     FREQUENCY = ...  # Frequency statistics. The number of occurrences of each index value is counted.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, hist_type=pl.HistType.FREQUENCY)
```
