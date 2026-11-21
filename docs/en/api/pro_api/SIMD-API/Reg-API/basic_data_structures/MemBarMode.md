# MemBarMode

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

MemBarMode defines the source-to-destination ordering constraints for the memory barrier in [vf.mem_bar](../data_movement/mem_bar.md). It is used to ensure that specified prior memory access operations complete before subsequent memory access operations begin.

## Prototype

```python
class MemBarMode(enum.Enum):
     VST_VLD = ...  # Vector Store -> Vector Load (RAW, default)
     VLD_VST = ...  # Vector Load -> Vector Store (WAR)
     VST_VST = ...  # Vector Store -> Vector Store (WAW)
     VST_LD = ...  # Vector Store -> Scalar Load
     VST_ST = ...  # Vector Store -> Scalar Store
     VLD_ST = ...  # Vector Load -> Scalar Store
     ST_VLD = ...  # Scalar Store -> Vector Load
     ST_VST = ...  # Scalar Store -> Vector Store
     LD_VST = ...  # Scalar Load -> Vector Store
     VV_ALL = ...  # All vectors -> All vectors
     VS_ALL = ...  # All vectors -> All scalars
     SV_ALL = ...  # All scalars -> All vectors
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD) # Ensure that Vector Store completes before Vector Load.
```
