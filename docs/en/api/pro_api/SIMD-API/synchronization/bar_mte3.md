# pypto_pro.language.system.bar_mte3

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

Performs barrier synchronization in the MTE3 pipeline and waits for the completion of operations previously issued in the MTE3 pipeline.

## Prototype

```python
pypto_pro.language.system.bar_mte3() -> None
```

## Parameters

None

## Constraints

- It can be called in the Cube or Vector section.
- It only waits for the completion of operations previously issued in the MTE3 pipeline within the current AI Core and does not perform cross-core synchronization.
- When the GM addresses written by consecutive MTE3 store operations overlap, call this API between the two operations to ensure that the previous store completes before the next store is executed. Otherwise, incorrect data may be produced.

## Return Value

None

## Examples

```python
with pl.section_vector():
    # ... MTE3 pipeline operations
    pl.system.bar_mte3()
    # ... Subsequent operations
```
