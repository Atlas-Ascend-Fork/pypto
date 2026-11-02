# pypto.reset\_options

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:28:54.463Z pushedAt=2026-10-28T11:57:29.581Z nl=2 -->

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

Resets all configuration items to their default values, including **codegen_options**, **host_options**, **pass_options**, **runtime_options**, and **verify_options**.

## Prototype

```python
reset_options() -> None
```

## Parameters

None

## Return Value

No return value. The setting takes effect immediately upon successful operation.

## Constraints

None

## Examples

```python
pypto.reset_options()
```
