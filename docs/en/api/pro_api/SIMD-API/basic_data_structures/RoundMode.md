# pypto_pro.language.RoundMode

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

Enumerates the rounding modes used by type conversion APIs. The rounding modes supported by different APIs and data type conversion combinations may vary. For details, see the parameter range or constraints of the corresponding API.

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

| Enumerated Value| Description| Example|
|---|---|---|
| CAST_NONE | The rounding mode is not explicitly specified. The specific conversion behavior is defined by the API that uses this enumeration.| - |
| CAST_RINT | Rounds to the nearest value, and rounds to the nearest even value in case of a tie.| 2.5 → 2, 3.5 → 4|
| CAST_ROUND | Rounds to the nearest value, and rounds away from zero in case of a tie.| 2.5 → 3, -2.5 → -3|
| CAST_FLOOR | Rounds towards negative infinity.| 1.6 → 1, -1.6 → -2|
| CAST_CEIL | Rounds towards positive infinity.| 1.6 → 2, -1.6 → -1|
| CAST_TRUNC | Rounds towards zero.| 1.6 → 1, -1.6 → -1|
| CAST_ODD | Sets the least significant bit of the result to 1 when precision loss occurs.| DT_FP32 to DT_FP16: 1.0001 → 1.0009765625|

## Constraints

- The default rounding mode of each API may vary.
- Only some enumerated values are supported for different data type conversion combinations. Before use, see the corresponding API document.
