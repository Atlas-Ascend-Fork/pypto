# pypto_pro.language.simt.warp_active_mask

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

Obtains the active lane set of the warp at the current instruction position and encodes it into a 32-bit unsigned integer. This API does not check service conditions. Instead, it directly records which lanes are executing the current instruction.

Each bit in the return value corresponds to a lane.

- The ith bit corresponds to lane i.
- If lane i is executing the current instruction, the ith bit is 1. Otherwise, the ith bit is 0.
- Active lanes that execute the same call site obtain the same bitmap.

The active lane set is related to the call site. When branch divergence occurs, each branch path contains only the lanes that are currently executing the path. After the paths converge, the lanes that re-execute the same instruction are included in the active set again.

## Prototype

```python
pypto_pro.language.simt.warp_active_mask() -> Scalar
```

## Parameters

None

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns the active lane bitmap of the DT_UINT32 type. If bit i is 1, lane i is active at the current instruction position. Active lanes that execute the same call site return the same result.

## Examples

In the following example, the first half and the second half of the warp consisting of 32 lanes enter different branches. The lanes are arranged in the order from bit 31 to bit 0. The first half of lanes obtain 00000000 00000000 11111111 11111111 when executing the first call site, and the second half of lanes obtain 11111111 11111111 00000000 00000000 when executing the second call site.

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_active_mask_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    if lane < pl.simt.warp_size() // 2:
        output[0, tid] = pl.simt.warp_active_mask()
        output[1, tid] = 0
    else:
        output[0, tid] = 0
        output[1, tid] = pl.simt.warp_active_mask()


@pl.jit()
def warp_active_mask_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        warp_active_mask_example[WARP_SIZE](output)
```
