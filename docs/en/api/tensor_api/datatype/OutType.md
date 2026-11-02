# OutType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:47:33.896Z pushedAt=2026-10-28T11:57:29.666Z nl=2 -->

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

**OutType** defines the type of output data. It specifies the output format of certain operations (such as comparison operations), distinguishing between Boolean value output and bit value output.

## Prototype

```python
class OutType(enum.Enum):
     BOOL = ...  # Boolean output type, which outputs True or False.
     BIT = ...   # Bit output type, which outputs a bit value of 0 or 1.
```
