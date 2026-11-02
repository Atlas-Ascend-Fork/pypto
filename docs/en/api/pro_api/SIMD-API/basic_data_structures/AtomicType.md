# pypto_pro.language.AtomicType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:42:24.552Z pushedAt=2026-10-28T11:57:28.589Z nl=2 -->

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

Enumeration of GM write modes, used to specify whether normal write or atomic add write is used when writing back to GM.

## Prototype

```python
PYPTO_DECLARE_ENUM(AtomicType,
    AtomicNone,   # Normal write. The original data in GM is overwritten by the source Tile data without accumulation.
    AtomicAdd     # Atomic add write. Accumulates the specified data into GM. Initialize the GM destination area before calling.
)
```

For supported data types and usage constraints, see [pypto_pro.language.store](../memory_data_movement/store.md#constraints).
