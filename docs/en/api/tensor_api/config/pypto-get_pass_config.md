# pypto.get\_pass\_config

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

Obtains the configuration information of the specified Pass.

## Prototype

```python
get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool
```

## Parameters

| Parameter         | Input/Output| Description                                                                |
|-----------------|-----------|----------------------------------------------------------------------|
| strategy        | Input     | Pass strategy name, for example, "PVC2_OOO".|
| identifier      | Input     | Pass name, for example, "ExpandFunction".|
| key             | Input     | PassConfigKey enumeration.<br> KEY_DUMP_GRAPH: Dumps the Pass computational graph.|
| default_value   | Input     | If the configuration value of the specified key is not found, the default value is returned.|

## Return Value

Returns the configuration value of key in the Pass whose name is identifier in the strategy.

## Constraints

key can only accept the specified enumeration values.

## Examples

```python
pypto.get_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
```
