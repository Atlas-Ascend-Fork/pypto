# pypto_pro.language.CacheLine

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

Enumerates the DCCI cache operation scope. It is used to specify whether to clean and invalidate a single cache line or the entire data cache.

## Prototype

```python
PYPTO_DECLARE_ENUM(CacheLine,
    SINGLE_CACHE_LINE,   # 64-byte cache line where the target valid address is located. The address does not need to be manually aligned.
    ENTIRE_DATA_CACHE    # Performs operations on the entire data cache.
)
```

For details about the constraints, see [dcci](../cache_control/dcci.md).
