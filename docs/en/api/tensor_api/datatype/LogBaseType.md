# LogBaseType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:45:49.515Z pushedAt=2026-10-28T11:57:29.663Z nl=2 -->

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

**LogBaseType** defines the base type for logarithmic operations, used to specify the base for logarithmic functions. It supports common logarithmic operations including natural logarithm, base-2 logarithm, and base-10 logarithm.

## Prototype

```python
class LogBaseType(enum.Enum):
     LOG_E = ...   # Natural logarithm base, used for logarithmic operations with base approximately 2.718.
     LOG_2 = ...   # Logarithmic operation with base 2, commonly used in information theory and computer science.
     LOG_10 = ...  # Logarithmic operation with base 10, commonly used in scientific computing and engineering applications.
```
