# DataCopyMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:23:46.365Z pushedAt=2026-10-28T11:57:28.839Z nl=2 -->

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

DataCopyMode defines the data copy granularity mode for [vf.load_align](../data_movement/load_align.md), [vf.store_align](../data_movement/store_align.md), and [vf.gather](../data_movement/gather.md), controlling whether data is copied element by element or by DataBlock.

## Prototype

```python
class DataCopyMode(enum.Enum):
     NORM = ...  # Normal element-by-element copy (default).
     DATA_BLOCK_LOAD = ...  # DataBlock load (for vf.gather: gather at 32B DataBlock granularity).
     DATA_BLOCK_COPY = ...  # Non-contiguous DataBlock copy (based on block_stride).
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY)
```
