# pypto.Tensor.get\_cache\_policy

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:07:20.926Z pushedAt=2026-10-28T11:57:30.308Z nl=2 -->

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

Obtains whether a cache policy is enabled.

## Prototype

```python
get_cache_policy(self, policy: CachePolicy) -> bool
```

## Parameters

| Parameter | Input/Output | Description |
|---------|-----------|----------------------------------------------------------------------|
| policy | Input | Cache policy type. |

## Return Value

Whether the cache policy is enabled.

## Constraints

None

## Examples

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
out = t.get_cache_policy(pypto.CachePolicy.PREFETCH)
```

The results are as follows:

```python
Output data out: False
```
