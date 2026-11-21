# CastMode

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

CastMode defines the rounding mode during data type conversion. It is used to control the precision handling during floating-point number conversions to ensure the accuracy of the conversion results.

## Prototype

```python
class CastMode(enum.Enum):
     CAST_NONE = ...   # No rounding mode is specified. The framework automatically selects the default value.
     CAST_RINT =...   # Round to the nearest integer. In case of a tie, rounds to the even integer (IEEE 754 by default).
     CAST_ROUND =...  # Round to the nearest integer. In case of a tie, rounds away from zero.
     CAST_FLOOR =...  # Round down, towards negative infinity.
     CAST_CEIL =...   # Round up, towards positive infinity.
     CAST_TRUNC =...  # Round towards zero.
     CAST_ODD =...    # Round to the nearest odd integer, which means the Von Neumann rounding.
```

## Rounding Mode Details

- Currently, CAST_ODD supports only conversion from FP32 to FP16.
- CAST_NONE is not a rounding mode supported by hardware, but a concept at the framework level. If no rounding mode is specified, the framework automatically selects the default value.

## Usage Description

### Default CastMode Rules

If the user uses CAST_NONE or specifies a rounding mode (CastMode) that is not supported by hardware, the framework automatically uses the following default rounding mode (default CastMode).

| Conversion Type| Default CastMode|
|----------|---------------|
| From a floating-point number to an integer (FP→INT)| CAST_TRUNC (rounding towards zero)|
| Other conversion scenarios| CAST_RINT (rounding to the nearest even)|

### Type Extension Conversion

Type extension conversion (such as from FP16 to FP32, from INT8 to FP16, or from INT16 to INT32) is lossless and does not require rounding. In this case, the CastMode parameter passed by the user is ignored.

### Handling of Unsupported CastMode

When the user specifies a rounding mode (CastMode) that is not supported by the hardware for a conversion:

1. The framework does not report an error.
2. The framework automatically uses the default CastMode of the conversion (complying with the preceding default rules).

## Description of Binary Rounding Rules

Before understanding the precision conversion rules, you need to know the basics about the representation modes of floating-point numbers and the rounding rules of binary numbers.

### Representation Modes of Floating-Point Numbers

- **DT_FP16**: There are 16 bits, including 1 sign bit (S), 5 exponent bits (E), and 10 mantissa bits (M).
  - When not all exponent bits (E) are 0 or 1, the value is: (-1)^S × 2^(E-15) × (1 + M).
  - When all exponent bits (E) are 0, the value is: (-1)^S × 2^(-14) × M.
  - When all exponent bits (E) are 1 and all mantissa bits (M) are 0, the value is ±inf (depending on the sign bit). When all exponent bits (E) are 1 and all mantissa bits (M) are not 0, the value is NaN.

- **DT_FP32**: There are 32 bits, including 1 sign bit (S), 8 exponent bits (E), and 23 mantissa bits (M).
  - When not all exponent bits (E) are 0 or 1, the value is: (-1)^S × 2^(E-127) × (1 + M).
  - When all exponent bits (E) are 0, the value is: (-1)^S × 2^(-126) × M.
  - When all exponent bits (E) are 1 and all mantissa bits (M) are 0, the value is ±inf (depending on the sign bit). When all exponent bits (E) are 1 and all mantissa bits (M) are not 0, the value is NaN.

- **DT_BF16**: There are 16 bits, including 1 sign bit (S), 8 exponent bits (E), and 7 mantissa bits (M).
  - When not all exponent bits (E) are 0 or 1, the value is: (-1)^S × 2^(E-127) × (1 + M).
  - When all exponent bits (E) are 0, the value is: (-1)^S × 2^(-126) × M.
  - When all exponent bits (E) are 1 and all mantissa bits (M) are 0, the value is ±inf (depending on the sign bit). When all exponent bits (E) are 1 and all mantissa bits (M) are not 0, the value is NaN.

### Behavior of Each Rounding Mode

- **CAST_RINT**: If the first bit of the portion to be rounded is 0, no carry is performed. If the first bit is 1 and not all following bits are 0, carry is performed. If the first bit is 1 and all following bits are 0: No carry is performed when the last M bit is 0; Carry is performed when the last M bit is 1.

- **CAST_FLOOR**: If bit S is 0, no carry is performed. If bit S is 1 and all the bits of the portion to be rounded are 0, no carry is performed. In other cases, carry is performed.

- **CAST_CEIL**: If bit S is 1, no carry is performed. If bit S is 0 and all the bits of the portion to be rounded are 0, no carry is performed. In other cases, carry is performed.

- **CAST_ROUND**: If the first bit of the portion to be rounded is 0, no carry is performed. In other cases, carry is performed.

- **CAST_TRUNC**: No carry is performed.

- **CAST_ODD**: If all the bits of the portion to be rounded are 0, no carry is performed. If not all the bits of the portion to be rounded are 0: No carry is performed when the last M bit is 1; Carry is performed when the last M bit is 0.
