# pypto.Tensor.set\_cache\_policy

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

Sets the attributes of the tensor cache.

## Prototype

```python
set_cache_policy(self, policy: CachePolicy, value: bool) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| policy  | Input     | Cache policy type. The options are as follows:<br> - CachePolicy.NONE_CACHEABLE: The chip provides the L2 cache capability. However, for some operators, having the L2 cache is actually worse than not having it. After this policy is configured, the tensor will not pass through the L2 cache. The common scenarios are as follows:<br>   - For constants such as weight, if the operator reads from out only once without reuse, there is no need to use the L2 cache.<br>   - If the output shape is too large, the memory first used by the lower-layer operator is not the final output result of the upper-layer operator. In this case, using the L2 cache causes the lower-level operator to write back to out, resulting in performance deterioration.|
| value   | Input     | Whether to enable the cache policy.|

## Return Value

None

## Constraints

None

## Examples

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
t.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```
