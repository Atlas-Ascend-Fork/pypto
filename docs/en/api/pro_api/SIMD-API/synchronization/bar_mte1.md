# pypto_pro.language.system.bar_mte1

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

Performs barrier synchronization in the MTE1 pipeline and waits for the completion of operations previously issued in the MTE1 pipeline.

## Prototype

```python
pypto_pro.language.system.bar_mte1() -> None
```

## Parameters

None

## Constraints

- It can be called only in the Cube section.
- It only waits for the completion of operations previously issued in the MTE1 pipeline within the current AI Core and does not perform cross-core synchronization.

## Return Value

None

## Examples

```python
with pl.section_cube():
    # ... MTE1 pipeline operations
    pl.system.bar_mte1()
    # ... Subsequent operations
```
