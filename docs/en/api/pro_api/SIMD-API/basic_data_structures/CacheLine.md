# pypto_pro.language.CacheLine

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:42:54.797Z pushedAt=2026-10-28T11:57:28.596Z nl=2 -->

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

Enumeration of DCCI cache operation scopes, used to specify whether to perform clean and invalidate operations on a single cache line or the entire data cache.

## Prototype

```python
PYPTO_DECLARE_ENUM(CacheLine,
    SINGLE_CACHE_LINE,   # The 64-byte cache line where the target effective address resides. The address does not need to be manually aligned.
    ENTIRE_DATA_CACHE    # Operate on the entire data cache.
)
```

For specific usage constraints, see [dcci](../cache_control/dcci.md).
