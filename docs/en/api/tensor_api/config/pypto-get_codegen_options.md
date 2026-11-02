# pypto.get\_codegen\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:21:23.055Z pushedAt=2026-10-28T11:57:29.535Z nl=2 -->

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

Obtains the **codegen** configuration.

## Prototype

```python
get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a **dict** containing all configuration items of **codegen**.

## Constraints

None

## Examples

```python
pypto.get_codegen_options()
```
