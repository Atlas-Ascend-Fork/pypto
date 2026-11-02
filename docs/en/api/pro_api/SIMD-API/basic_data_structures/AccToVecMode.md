# pypto_pro.language.AccToVecMode

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:42:37.044Z pushedAt=2026-10-28T11:57:28.593Z nl=2 -->

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

An enum for dual-target control. It is an important attribute of [pypto_pro.language.move](../memory_data_movement/move.md) and is used in the L0C Buffer->UB data movement scenario.

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

| Parameter Value | Description |
|:-------|:-----|
| SingleModeVec0 | Single-target mode, writes the entire matrix to the target UB of Vec0. |
| SingleModeVec1 | Single-target mode, writes the entire matrix to the target UB of Vec1. |
| DualModeSplitM | Dual-target mode, splits along the M dimension, writing M/2*N elements to each UB.<br>In the tail block scenario, the framework **automatically** aligns valid_M (the tail block size of the L0C Buffer along the M axis) upward to a multiple of 2 to obtain aligned_M before the data movement. You only need to understand the splitting strategy on the UB side: Vec0 (sub_id=0) receives the first aligned_M / 2 rows, and Vec1 (sub_id=1) receives the remaining valid_M - aligned_M / 2 rows. For details, see [Examples](#tail-block-scenario-in-dualmodesplitm-mode). **Note**: When valid_M is 1, the data is split only to Vec0. |
| DualModeSplitN | Dual-target mode, splits along the N dimension, writing M*N/2 elements to each UB.<br>In the tail block scenario, the framework **automatically** aligns valid_N (the tail block size of the L0C Buffer along the N axis) upward to a multiple of 32 to obtain aligned_N before the data movement. You only need to understand the splitting strategy on the UB side: Vec0 (sub_id=0) receives the first aligned_N / 2 columns, and Vec1 (sub_id=1) receives the remaining valid_N - aligned_N / 2 columns. For details, see [Examples](#dualmodesplitn-mode-tail-block-scenario). **Note**: When valid_N does not exceed 16, the data is split only to Vec0. |

## Examples

### Tail Block Scenario in DualModeSplitM Mode

```python
import pypto_pro.language as pl

# Cube execution domain: You only need to set the actual valid_M, without manual alignment.
pl.set_validshape(ac, [valid_M, N])       # valid_M=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)  # The framework aligns automatically, aligned_M=34.

# Vector execution domain: You need to calculate the actual UB sizes in Vec0/Vec1 yourself.
v0 = (valid_M + 1) // 2 * 2 // 2          # v0 = 17
v1 = valid_M - v0                         # v1 = 16
if sub_id == 0:
    pl.set_validshape(vec, [v0, N])
else:
    pl.set_validshape(vec, [v1, N])
```

### DualModeSplitN Mode Tail Block Scenario

```python
import pypto_pro.language as pl

# Cube execution domain: you only need to set the actual valid_N, without manual alignment.
pl.set_validshape(ac, [TILE, valid_N])     # valid_N=33
pl.matmul(ac, al, br)
pl.move(vec, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)  # The framework aligns automatically, aligned_N=64.

# Vector execution domain: on the V side, you compute v0/v1 yourself.
v0 = (valid_N + 31) // 32 * 32 // 2        # v0 = 32
v1 = valid_N - v0                          # v1 = 1
if sub_id == 0:
    pl.set_validshape(vec, [TILE, v0])
else:
    pl.set_validshape(vec, [TILE, v1])
```
