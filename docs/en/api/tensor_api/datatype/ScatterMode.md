# ScatterMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:50:31.776Z pushedAt=2026-10-28T11:57:29.689Z nl=2 -->

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

**ScatterMode** defines the reduction mode of the **scatter** function.

## Prototype

```python
class ScatterMode(enum.Enum):
     NONE = ...     # Data movement only.
     ADD = ...      # Addition mode.
     MULTIPLY = ... # Multiplication mode.
```
