# pypto_pro.language.system.bar_mte1

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:25:46.721Z pushedAt=2026-10-28T11:57:29.160Z nl=2 -->

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

Performs barrier synchronization in the MTE1 pipeline, waiting for previously issued operations in the MTE1 pipeline to complete.

## Prototype

```python
pypto_pro.language.system.bar_mte1() -> None
```

## Parameters

None

## Constraints

- Only supported for invocation in the Cube section.
- Only waits for previously issued operations in the MTE1 pipeline within the current AI Core; it does not perform cross-core synchronization.

## Return Value

None

## Examples

```python
with pl.section_cube():
    # ... MTE1 pipeline operations
    pl.system.bar_mte1()
    # ... Subsequent operations
```
