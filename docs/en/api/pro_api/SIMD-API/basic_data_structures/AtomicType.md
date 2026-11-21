# pypto_pro.language.AtomicType

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

Enumerates GM write modes. It is used to specify whether to use the normal write or atomic add write mode when writing data back to the GM.

## Prototype

```python
PYPTO_DECLARE_ENUM(AtomicType,
    AtomicNone,   # Normal write. The original data in the GM is overwritten by the source tile data and is not accumulated.
    AtomicAdd     # Atomic add write. The specified data is accumulated to the GM. Before calling this API, you must initialize the GM destination region.
)
```

For details about the supported data types and usage restrictions, see [pypto_pro.language.store](../memory_data_movement/store.md#Constraints).
