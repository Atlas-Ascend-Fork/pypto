# LoadDist

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

LoadDist defines the data load distribution mode for [vf.load_align](../data_movement/load_align.md). It controls how data is moved from UB to registers. Different modes correspond to different hardware instructions and affect the data layout in registers as well as the broadcast/sampling behavior.

## Prototype

```python
class LoadDist(enum.Enum):
     # RegTensor target
     NORM = ...  # Normal element-wise load (default).
     BRC = ...  # Broadcast a single element to the entire register.
     BRC_B8 = ...  # Broadcast with 8-bit element width.
     BRC_B16 = ...  # Broadcast with 16-bit element width.
     BRC_B32 = ...  # Broadcast with 32-bit element width.
     US = ...  # Upsample (each element repeated twice).
     US_B8 = ...  # Upsample with 8-bit element width.
     US_B16 = ...  # Upsample with 16-bit element width.
     DS = ...  # Downsample (discarding every other element).
     DS_B8 = ...  # Downsample with 8-bit element width.
     DS_B16 = ...  # Downsample with 16-bit element width.
     UNPK =...  # Unpack.
     UNPK_B8 = ...  # Unpack with 8-bit element width.
     UNPK_B16 = ...  # Unpack with 16-bit element width.
     UNPK_B32 = ...  # Unpack with 32-bit element width.
     UNPK4 = ...  # 4-element unpacking.
     BLK =...  # Block copy.
     E2B = ...  # 16-bit to 32-bit width expansion.
     E2B_B16 = ...  # Expand with 16-bit element width.
     E2B_B32 = ...  # Expand with 32-bit element width.
     DINTLV_B8 = ...  # Deinterleave with 8-bit element width (split even/odd registers).
     DINTLV_B16 = ...  # Deinterleave with 16-bit element width.
     DINTLV_B32 = ...  # Deinterleave with 32-bit element width.
     # MaskReg target
     # NORM/US/DS are the same as for the RegTensor target; the transferred data volume is VL/8 bytes.
```

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    reg = vf.load_align(ub_tile, dist=pl.LoadDist.BRC)
```
