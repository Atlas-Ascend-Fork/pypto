# pypto_pro.language.system.bar_mte2

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

Performs barrier synchronization in the MTE2 pipeline and waits for the completion of operations previously issued in the MTE2 pipeline.

## Prototype

```python
pypto_pro.language.system.bar_mte2() -> None
```

## Parameters

None

## Constraints

- It can be called in the Cube or Vector section.
- It only waits for the completion of operations previously issued in the MTE2 pipeline within the current AI Core and does not perform cross-core synchronization.
- When the on-chip addresses written by consecutive MTE2 load operations overlap, call this API between the two operations to ensure that the previous load completes before the next load is executed. Otherwise, incorrect data may be produced.

## Return Value

None

## Examples

```python
with pl.section_vector():
    # ... MTE2 pipeline operations
    pl.system.bar_mte2()
    # ... Subsequent operations
```
