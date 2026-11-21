# pypto.get\_debug\_options

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
<!-- npu="9030" id5 -->
- Kirin 9030 processors: Supported
<!-- end id5 -->
<!-- npu="x90" id4 -->
- Kirin X90 processors: Supported
<!-- end id4 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

Obtains the debug configuration.

## Prototype

```python
get_debug_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None

## Return Value

Returns a dict containing all debug configuration items.

## Constraints

None

## Examples

```python
pypto.get_debug_options()
```
