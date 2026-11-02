# pypto.Tensor.set\_cache\_policy

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:18:49.746Z pushedAt=2026-10-28T11:57:30.379Z nl=2 -->

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

Sets the attributes of the tensor cache.

## Prototype

```python
set_cache_policy(self, policy: CachePolicy, value: bool) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| policy  | Input      | Cache policy type. The options are as follows: <br> - CachePolicy.NONE_CACHEABLE: The chip provides L2 Cache capability, but for some operators, having L2 Cache is worse than not having it due to their characteristics. After this policy is configured, the tensor does not go through L2 Cache. Common scenarios include: <br>   - For constants such as weight, if the operator reads from out only once without reuse, there is no need to enter L2. <br>   - If the output shape is too large and the memory first used by the lower-level operator is not the final output of the upper-level operator, entering L2 causes the lower-level operator to write back to out, degrading performance. |
| value   | Input      | Whether to enable this cache policy. |

## Return Value

None

## Constraints

None

## Examples

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
t.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```
