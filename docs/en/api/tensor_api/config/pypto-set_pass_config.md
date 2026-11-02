# pypto.set\_pass\_config

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:36:15.416Z pushedAt=2026-10-28T11:57:29.611Z nl=2 -->

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

Modifies the configuration information of a specified Pass.

## Prototype

```python
set_pass_config(strategy: str, identifier: str, key: PassConfigKey, value: bool)
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| strategy | Input | Pass strategy name, such as "PVC2_OOO". |
| identifier | Input | Pass name, such as "ExpandFunction". |
| key | Input | PassConfigKey enumeration. <br> KEY_DUMP_GRAPH: Dump the Pass computation graph. |
| value | Input | Configuration value. |

## Return Value

None

## Constraints

- Timing: This function must be called before graph compilation starts.
- Scope: The configuration is global and affects all subsequent compilation processes.

## Examples

```python
pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
