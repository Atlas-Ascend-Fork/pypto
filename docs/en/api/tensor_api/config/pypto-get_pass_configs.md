# pypto.get\_pass\_configs

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:25:59.055Z pushedAt=2026-10-28T11:57:29.568Z nl=2 -->

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

Obtains all configuration information of a specified Pass.

## Prototype

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|----------------|--------------|------------------------------------------------------------------------------|
| strategy       | Input        | Pass strategy name, for example, "PVC2_OOO". |
| identifier     | Input        | Pass name, for example, "ExpandFunction". |

## Return Value

A **PassConfigs** object containing the following read-only attributes:

| Attribute         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| printGraph         | Dumps the computation graph IR. |
| dumpGraph          | Dumps the computation graph. |
| dumpPassTimeCost   | Dumps the Pass execution time. |
| preCheck           | Performs validation before Pass execution. |
| postCheck          | Performs validation after Pass execution. |
| disablePass        | Skips the current Pass. |
| healthCheck        | Performs a health check and generates a report. |

## Constraints

None

## Examples

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```
