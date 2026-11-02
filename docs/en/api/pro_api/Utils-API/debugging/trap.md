# pypto_pro.language.trap

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:15:34.836Z pushedAt=2026-10-28T11:57:29.501Z nl=2 -->

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

Forcefully aborts Kernel execution for debugging purposes. The abort is unconditional and accepts no parameters.

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
# Conditional abort.
if flag:
    pl.trap()

# Unconditional abort (for debugging).
pl.trap()
```
