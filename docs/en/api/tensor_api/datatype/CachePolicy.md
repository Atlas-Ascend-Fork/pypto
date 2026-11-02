# CachePolicy

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:43:24.036Z pushedAt=2026-10-28T11:57:29.647Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->

## Description

**CachePolicy** defines the tensor cache policy, which is used to control the behavior of data in caches at various levels, optimize memory access performance, and reduce memory bandwidth consumption.

## Prototype

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # Prefetch policy. Loads data into the cache in advance to reduce access latency.
     NONE_CACHEABLE = ...  # Non-cacheable policy. Data is not stored in the cache and is accessed directly from main memory.
```
