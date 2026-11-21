# pypto_pro.language.DcciDst

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

Enumerates DCCI consistency targets. It is used to specify the storage area or hardware access path that needs to be consistent with the data cache.

## Prototype

```python
PYPTO_DECLARE_ENUM(DcciDst,
    AUTO,             # Automatic selection based on the target object: CACHELINE_OUT for GM tensors, and CACHELINE_UB for UB tiles.
    CACHELINE_OUT,    # Ensures the consistency between the data cache and the GM, and is applicable to GM tensors.
    CACHELINE_UB,     # Ensures the consistency between the data cache and the UB, and is applicable to UB tiles.
    CACHELINE_ALL,    # Has the same effect as CACHELINE_OUT, and is applicable to GM tensors.
    CACHELINE_ATOMIC  # Ensures the consistency between the data cache and the GM in the hardware atomic cache path.
)
```

When an enumerated value is explicitly specified, it must match the storage area and hardware access path of the target object. For details about the constraints, see [dcci](../cache_control/dcci.md).
