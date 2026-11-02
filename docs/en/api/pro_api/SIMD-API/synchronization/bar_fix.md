# pypto_pro.language.system.bar_fix

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:23:07.909Z pushedAt=2026-10-28T11:57:29.154Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Performs barrier synchronization in the FIX pipeline, waiting for the previously issued operations in the FIX pipeline to complete.

## Prototype

```python
pypto_pro.language.system.bar_fix() -> None
```

## Parameters

None

## Constraints

- It can be called only in the Cube section.
- It waits only for the previously issued operations in the FIX pipeline within the current AI Core and does not perform cross-core synchronization.

## Return Value

None

## Examples

```python
with pl.section_cube():
    # ... FIX pipeline operations
    pl.system.bar_fix()
    # ... Subsequent operations
```
