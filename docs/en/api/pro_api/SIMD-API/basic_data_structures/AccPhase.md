# pypto_pro.language.AccPhase

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

An enumeration for enabling the hardware unitFlag mechanism (fine-grained parallelism between Mmad and Fixpipe instructions). The hardware provides a unit-flag bit for each 512-byte memory block in the L0C Buffer to indicate whether the memory block can be read or written. In this way, the read and write operations of the L0C Buffer memory are synchronized at the hardware level.
For matrix multiplication, when the unit-flag bit is 0, data can be written to the L0C Buffer. Otherwise, the write operations are blocked until the unit-flag bit becomes 0. This API must be used together with the [pypto_pro.language.STPhase](STPhase.md) configuration of L0C Buffer move-out API ([pypto_pro.language.store](../memory_data_movement/store.md), [pypto_pro.language.store_tile](../memory_data_movement/store_tile.md), or [pypto_pro.language.move](../memory_data_movement/move.md)). Otherwise, the system may be hung.

## Prototype

```python
PYPTO_DECLARE_ENUM(AccPhase,
    Unspecified,  # The unitFlag mechanism is disabled.
    Partial,      # Only the unit-flag bit is checked, and write operations are controlled based on the 0/1 status. After the instructions are executed, the unit-flag bit of the memory block remains unchanged.
    Final         # The unit-flag bit is checked, and write operations are controlled based on the 0/1 status. After the instructions are executed, the unit-flag bit of the memory block is set to 1.
)
```
