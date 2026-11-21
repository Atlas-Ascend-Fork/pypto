# pypto_pro.language.MemDsb

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

Enumerates the data synchronization barrier (DSB) scope. It is used to specify the memory access range for `pl.system.dsb` sorting. The enumerated values correspond to DSB_ALL, DSB_DDR, and DSB_UB in the toolchain `mem_dsb_t` (cce_aicore_intrinsics.h). During code generation, DSB_ prefix is added to map to the corresponding hardware instructions.

## Prototype

```python
PYPTO_DECLARE_ENUM(MemDsb,
    ALL, # Sorts all memory access instructions (default).
    DDR,  # Sorts only memory access instructions that access the DDR.
    UB,   # Sorts only memory access instructions that access the UB.
)
```

When an enumerated value is explicitly specified, it must match the actual memory access path to be sorted. Use the DDR only when you need to ensure that GM writes are visible. For details about the constraints, see [dsb](../synchronization/dsb.md).
