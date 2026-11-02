# ReLuType

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:49:13.934Z pushedAt=2026-10-28T11:57:29.683Z nl=2 -->

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

**ReLuType** defines the modes of the ReLU activation function, used to enable the ReLU function.

## Prototype

```python
class ReLuType(enum.Enum):
     NO_RELU= ...  # Disable the ReLu function.
     RELU= ...     # Enable the ReLu function.
```
