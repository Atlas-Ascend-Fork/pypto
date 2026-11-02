# pypto.get\_pass\_config

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:25:56.164Z pushedAt=2026-10-28T11:57:29.562Z nl=2 -->

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

Obtains the configuration information of the specified **Pass**.

## Prototype

```python
get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool
```

## Parameters

| Parameter       | Input/Output | Description                                                          |
|-----------------|--------------|----------------------------------------------------------------------|
| strategy        | Input        | Pass strategy name, for example, "PVC2_OOO". |
| identifier      | Input        | Pass name, for example, "ExpandFunction". |
| key             | Input        | PassConfigKey enumeration. <br> KEY_DUMP_GRAPH dumps the Pass computation graph. |
| default_value   | Input        | Default value returned if the configuration value of the specified key is not found. |

## Return Value

Returns the configuration value of **key** in the Pass named **identifier** under the specified **strategy**.

## Constraints

**key** can only accept the specified enumeration values.

## Examples

```python
pypto.get_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
```
