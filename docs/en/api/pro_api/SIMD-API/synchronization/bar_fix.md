# pypto_pro.language.system.bar_fix

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

Performs barrier synchronization in the FIX pipeline and waits for the completion of operations previously issued in the FIX pipeline.

## Prototype

```python
pypto_pro.language.system.bar_fix() -> None
```

## Parameters

None

## Constraints

- It can be called only in the Cube section.
- It only waits for the completion of operations previously issued in the FIX pipeline within the current AI Core and does not perform cross-core synchronization.

## Return Value

None

## Examples

```python
with pl.section_cube():
    # ... FIX pipeline operations
    pl.system.bar_fix()
    # ... Subsequent operations
```
