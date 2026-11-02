# pypto_pro.language.AccPhase

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:41:32.727Z pushedAt=2026-10-28T11:57:28.586Z nl=2 -->

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

An enumeration that enables the hardware unitFlag mechanism, a fine-grained parallelism between Mmad and Fixpipe instructions. The hardware provides a unit-flag for each 512B memory block in the L0C Buffer to indicate whether the block can be read or written, thereby implementing read/write synchronization of the L0C Buffer memory at the hardware level.
For matrix multiplication, when the unit-flag is 0, the L0C Buffer can be written to; otherwise, the write operation is blocked until the flag becomes 0. When used, it must be configured together with the STPhase configuration of the L0C Buffer data movement APIs ([pypto_pro.language.store](../memory_data_movement/store.md), [pypto_pro.language.store_tile](../memory_data_movement/store_tile.md), or [pypto_pro.language.move](../memory_data_movement/move.md)); otherwise, a hang may occur.

## Prototype

```python
PYPTO_DECLARE_ENUM(AccPhase,
    Unspecified,  # Do not enable the unitFlag mechanism.
    Partial,      # Only check the unit-flag and control the write operation based on its 0/1 state. After the instruction is executed, the unit-flag of the memory block is not changed.
    Final         # Check the unit-flag and control the write operation based on its 0/1 state. After the instruction is executed, set the unit-flag of the memory block to 1.
)
```
