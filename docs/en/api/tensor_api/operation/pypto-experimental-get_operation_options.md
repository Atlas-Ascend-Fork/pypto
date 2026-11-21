# pypto.experimental.get\_operation\_options

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

Obtains the operation configuration.

## Prototype

```python
get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all configuration items of the operation.

## Constraints

1. The return value is of the Dict type and contains all configuration items of the operation.
2. Different configuration items may have different types, such as str, int, List[int], and Dict[int, int].

## Examples

```python
pypto.experimental.get_operation_options()
```
