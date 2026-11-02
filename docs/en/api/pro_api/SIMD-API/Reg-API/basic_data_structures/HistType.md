# HistType

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:24:35.100Z pushedAt=2026-10-28T11:57:28.849Z nl=2 -->

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

HistType defines the statistical mode of the [vf.histograms](../sorting_and_indexing/histograms.md) histogram, specifying whether to use accumulation statistics or frequency statistics.

## Prototype

```python
class HistType(enum.Enum):
     ACCUMULATE = ...  # Accumulation mode (default). The statistical result is accumulated onto the existing value in the target register.
     FREQUENCY = ...  # Frequency mode. Counts the occurrences of each index value.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.histograms(src, preg, hist_type=pl.HistType.FREQUENCY)
```
