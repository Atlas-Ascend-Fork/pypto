# pypto_pro.language.system.bar_mte3

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:26:34.842Z pushedAt=2026-10-28T11:57:29.170Z nl=2 -->

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

Performs barrier synchronization in the MTE3 pipeline and waits for the previously issued operations in the MTE3 pipeline to complete.

## Prototype

```python
pypto_pro.language.system.bar_mte3() -> None
```

## Parameters

None

## Constraints

- Can be called in the Cube section or the Vector section.
- Waits only for the previously issued operations in the MTE3 pipeline of the current AI Core, without performing cross-core synchronization.
- When the GM addresses written by consecutive MTE3 load operations overlap, call this API between the two operations to ensure that the previous load completes before the next load is executed. Otherwise, incorrect data may be produced.

## Return Value

None

## Examples

```python
with pl.section_vector():
    # ... MTE3 pipeline operations
    pl.system.bar_mte3()
    # ... Subsequent operations
```
