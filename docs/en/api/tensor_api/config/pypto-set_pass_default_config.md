# pypto.set\_pass\_default\_config

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:35:15.941Z pushedAt=2026-10-28T11:57:29.607Z nl=2 -->

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

Modifies the default configuration of a Pass. Its main function is to dynamically modify the runtime behavior configuration of a Pass. Currently, it supports configuring the switch for dumping computation graphs to facilitate analysis and debugging.

## Prototype

```python
set_pass_default_config(key: PassConfigKey, value: bool)
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | Input      | PassConfigKey enumeration. <br> KEY_DUMP_GRAPH: Dump the Pass computation graph. |
| value   | Input      | Configuration value. |

## Return Value

None

## Constraints

- Timing: This function must be called before graph compilation starts.
- Scope: The configuration is global and affects all subsequent compilation processes.

## Examples

```python
pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```
