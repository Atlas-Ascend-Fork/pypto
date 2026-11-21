# pypto_pro.language.AccToVecMode

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

Enumerates dual-target control, which is an important attribute of [pypto_pro.language.move](../memory_data_movement/move.md), and is used in the scenario where data is moved from the L0C Buffer to the UB.

## Prototype

```python
PYPTO_DECLARE_ENUM(AccToVecMode,
     SingleModeVec0,
     SingleModeVec1,
     DualModeSplitM,
     DualModeSplitN
)
```

## Parameters

| Value| Description|
|:-------|:-----|
| SingleModeVec0 | Single-target mode. The entire matrix is written to the destination UB of Vec0.|
| SingleModeVec1 | Single-target mode. The entire matrix is written to the destination UB of Vec1.|
| DualModeSplitM | Dual-target mode. The matrix is split along the M dimension, and M/2 × N elements are written to each UB.<br>In the tail block scenario, the framework **automatically** aligns valid_M (the tail block size of the L0C buffer along the M direction) upward to a multiple of 2 to obtain aligned_M before the movement. You only need to understand the splitting strategy on the UB side: Vec0 (sub_id=0) obtains the first aligned_M/2 rows, and Vec1 (sub_id=1) obtains the remaining valid_M - aligned_M / 2 rows. For details, see [Examples](#tail-block-scenario-in-dualmodesplitm-mode). **Note**: When valid_M is 1, the data is only split to Vec0.|
| DualModeSplitN | Dual-target mode. The matrix is split along the N dimension, and M × N/2 elements are written to each UB.<br>In the tail block scenario, the framework **automatically** aligns valid_N (the tail block size of the L0C Buffer along the N direction) upward to a multiple of 32 to obtain aligned_N before the movement. You only need to understand the splitting strategy on the UB side: Vec0 (sub_id=0) obtains the first aligned_N/2 columns, and Vec1 (sub_id=1) obtains the remaining valid_N - aligned_N / 2 columns. For details, see [Examples](#tail-block-scenario-in-dualmodesplitn-mode). **Note**: When valid_N is no more than 16, the data is only split to Vec0.|

## Examples

### Tail Block Scenario in DualModeSplitM Mode

```python
import pypto_pro.language as pl

# Cube execution domain: You only need to set the actual valid_M. Manual alignment is not required.
pl.set_validshape(ac, [valid_M, N])       # valid_M=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # The framework automatically aligns the value, and aligned_M is 34.

# Vector execution domain: You need to compute the actual UB sizes in Vec0/Vec1 yourself.
v0 = (valid_M + 1) // 2 * 2 // 2          # v0 = 17
v1 = valid_M - v0                         # v1 = 16
if sub_id == 0:
    pl.set_validshape(vec, [v0, N])
else:
    pl.set_validshape(vec, [v1, N])
```

### Tail Block Scenario in DualModeSplitN Mode

```python
import pypto_pro.language as pl

# Cube execution domain: You only need to set the actual valid_N. Manual alignment is not required.
pl.set_validshape(ac, [TILE, valid_N])     # valid_N=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)  # The framework automatically aligns the value, and aligned_N is 64.

# Vector execution domain: You need to compute v0/v1 yourself.
v0 = (valid_N + 31) // 32 * 32 // 2        # v0 = 32
v1 = valid_N - v0                          # v1 = 1
if sub_id == 0:
    pl.set_validshape(vec, [TILE, v0])
else:
    pl.set_validshape(vec, [TILE, v1])
```
