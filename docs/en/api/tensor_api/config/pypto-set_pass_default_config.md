# pypto.set\_pass\_default\_config

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

Modifies the default configuration of a Pass. It is used to dynamically modify the runtime behavior configuration of a Pass. Currently, the dump computational graph switch can be configured to facilitate analysis and debugging.

## Prototype

```python
set_pass_default_config(key: PassConfigKey, value: bool)
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| key     | Input     | PassConfigKey enumeration.<br> KEY_DUMP_GRAPH: Dumps the Pass computational graph.|
| value   | Input     | Configured value|

## Return Value

None

## Constraints

- Setting time: This function must be called before graph build starts.
- Scope: The configuration is global and affects all subsequent compilation processes.

## Examples

```python
pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
