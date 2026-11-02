# pypto.experimental.get\_operation\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:34:08.720Z pushedAt=2026-10-28T11:57:29.883Z nl=2 -->

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

Obtains the **operation** configuration.

## Prototype

```python
get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict** containing all configuration items of the **operation**.

## Constraints

1. The return value is of the **Dict** type and contains all configuration items of the **operation**.
2. Different configuration items may have different types, such as **str**, **int**, **List[int]**, and **Dict[int, int]**.

## Examples

```python
pypto.experimental.get_operation_options()
```
