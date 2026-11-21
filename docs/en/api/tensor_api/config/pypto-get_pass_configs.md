# pypto.get\_pass\_configs

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

Obtains all configuration information of the specified Pass.

## Prototype

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## Parameters

| Parameter    | Input/Output| Description                                                                |
|------------|-----------|----------------------------------------------------------------------|
| strategy   | Input     | Pass strategy name, for example, "PVC2_OOO".|
| identifier | Input     | Pass name, for example, "ExpandFunction".|

## Return Value

PassConfigs object. The object has the read-only attributes listed below.

| Attribute              | Description                                                                |
|--------------------|----------------------------------------------------------------------|
| printGraph         | Dumps the computational graph IR.|
| dumpGraph          | Dumps the computational graph.|
| dumpPassTimeCost   | Dumps the time consumed by the Pass.|
| preCheck           | Performs a check before the Pass is executed.|
| postCheck          | Performs a check after the Pass is executed.|
| disablePass        | Skips the current Pass.|
| healthCheck        | Performs a health check and generates a report.|

## Constraints

None

## Examples

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```
