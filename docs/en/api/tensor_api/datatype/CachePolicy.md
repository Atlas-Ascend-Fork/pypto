# CachePolicy

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

CachePolicy defines the cache policy of a tensor. It is used to control the behavior of data in caches at various levels, optimize memory access performance, and reduce memory bandwidth consumption.

## Prototype

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # Prefetch policy. Data is loaded to the cache in advance to reduce access latency.
     NONE_CACHEABLE = ...  # Non-cacheable policy. Data is not stored in the cache and is directly accessed from the main memory.
```
