# ReduceMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:48:46.902Z pushedAt=2026-10-28T11:57:29.676Z nl=2 -->

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

**ReduceMode** defines the execution mode of reduction operations. It specifies the reduction computation method in multi-threaded or multi-device environments, ensuring the correctness and performance of computation results.

## Prototype

```python
class ReduceMode(enum.Enum):
     ATOMIC_ADD = ...  # Atomic addition reduction, using atomic operations to ensure thread-safe data accumulation.
```
