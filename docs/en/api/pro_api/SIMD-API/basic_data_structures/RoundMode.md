# pypto_pro.language.RoundMode

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:47:38.534Z pushedAt=2026-10-28T11:57:28.640Z nl=2 -->

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

Rounding mode enumeration used by type conversion APIs. Different APIs and data type conversion combinations may support different rounding modes. For details, see the parameter range or constraints of the corresponding API.

## Prototype

```python
class RoundMode(enum.Enum):
    CAST_NONE = ...
    CAST_RINT = ...
    CAST_ROUND = ...
    CAST_FLOOR = ...
    CAST_CEIL = ...
    CAST_TRUNC = ...
    CAST_ODD = ...
```

## Parameters

| Enum Value | Description | Example |
|---|---|---|
| CAST_NONE | No rounding rule is explicitly specified. The actual conversion behavior is defined by the API that uses this enumeration. | - |
| CAST_RINT | Round to the nearest value, with ties rounding to even. | 2.5 → 2, 3.5 → 4 |
| CAST_ROUND | Round to the nearest value, with ties rounding away from zero. | 2.5 → 3, -2.5 → -3 |
| CAST_FLOOR | Round toward negative infinity. | 1.6 → 1, -1.6 → -2 |
| CAST_CEIL | Round toward positive infinity. | 1.6 → 2, -1.6 → -1 |
| CAST_TRUNC | Round toward zero. | 1.6 → 1, -1.6 → -1 |
| CAST_ODD | When precision is lost, set the least significant bit of the result to 1. | DT_FP32 to DT_FP16: 1.0001 → 1.0009765625 |

## Constraints

- The default rounding mode may differ across APIs.
- Different data type conversion combinations support only some enum values. Check the corresponding API documentation before use.
