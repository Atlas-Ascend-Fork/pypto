# pypto.get\_pass\_default\_config

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:26:39.535Z pushedAt=2026-10-28T11:57:29.565Z nl=2 -->

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

Obtains the default configuration information of a Pass.

## Prototype

```python
get_pass_default_config(key: PassConfigKey, default_value: bool) -> bool
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| key             | Input      | PassConfigKey enumeration. <br> KEY_DUMP_GRAPH dumps the Pass computation graph. |
| default_value   | Input      | Default value returned if the configuration value for the specified key is not found. |

## Return Value

Returns the default configuration value of the Pass named **key**, or **default_value** if it does not exist.

## Constraints

**key** can only accept the specified enumeration values.

## Examples

```python
pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
