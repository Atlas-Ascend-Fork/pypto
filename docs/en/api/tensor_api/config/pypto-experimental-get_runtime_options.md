# pypto.experimental.get\_runtime\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:20:13.402Z pushedAt=2026-10-28T11:57:29.532Z nl=2 -->

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

Obtains the experimental configuration information of the runtime in the current config.

## Prototype

```python
get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a dict containing the experimental configuration items of the runtime.

## Constraints

1. The return value is of the Dict type, containing the runtime configuration items.
2. Different configuration items may have different types, such as str, int, List[int], and List[str].

## Examples

```python
pypto.experimental.get_runtime_options()
```
