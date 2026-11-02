# pypto.get\_verify\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:28:52.598Z pushedAt=2026-10-28T11:57:29.578Z nl=2 -->

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

Obtains the currently configured options of the precision debugging **Verify** feature.

## Prototype

```python
get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a dictionary containing the current settings of the precision debugging Verify feature.

## Constraints

## Examples

```python
pypto.get_verify_options()
```
