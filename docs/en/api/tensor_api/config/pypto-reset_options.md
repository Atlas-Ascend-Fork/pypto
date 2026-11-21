# pypto.reset\_options

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

Resets all configuration items to their default values, including codegen\_options, host\_options, pass\_options, runtime\_options, and verify\_options.

## Prototype

```python
reset_options() -> None
```

## Parameters

None

## Return Value

None: No value is returned. The setting takes effect immediately after the operation is successful.

## Constraints

None

## Examples

```python
pypto.reset_options()
```
