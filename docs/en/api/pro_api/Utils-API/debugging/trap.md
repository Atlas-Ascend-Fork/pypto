# pypto_pro.language.trap

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

Forcibly aborts kernel execution for debugging. This is an unconditional abort, which does not accept any parameter.

## Prototype

```python
pypto_pro.language.trap() -> None
```

## Parameters

None

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl
# Conditional abort
if flag:
    pl.trap()

# Unconditional abort (for debugging)
pl.trap()
```
