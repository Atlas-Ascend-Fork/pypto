# PrecisionType

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Not supported
<!-- end id11 -->

## Description

PrecisionType defines the precision mode of high-precision vector operators. It is used to control the precision handling during computation of division, modulus, exponent, and logarithm.

## Prototype

```python
class PrecisionType(enum.Enum):
    INTRINSIC = ...       # Instruction mode. Chip instructions are directly used.
    HIGH_PRECISION = ...  # High-precision mode
```

## Parameters

| Value| Description|
|:-------|:-----|
| HIGH_PRECISION | High-precision mode. A higher-precision computation method is used in the underlying implementation.|
| INTRINSIC | Instruction mode. Chip instructions are directly used for computation, resulting in higher performance.|

## Suggestions

1. **Default behavior**: If the precision mode is not specified, the high-precision mode (`HIGH_PRECISION`) is used by default to ensure the computation precision.
2. **Scenarios with high precision requirements**: `HIGH_PRECISION` is recommended, which can effectively reduce precision loss and improve the accuracy of the computation results.
3. **Scenarios with low precision requirements but high performance requirements**: The instruction mode (`INTRINSIC`) can be used, where the chip instructions are directly used for computation.
4. Usage of HIGH_PRECISION:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: Supported
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: Not supported
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: Not supported
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: Not supported
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: Not supported
   <!-- end id10 -->
   <!-- npu="9050" id12 -->
   - Kirin 9050 processors: Not supported
   <!-- end id12 -->

## Supported Operators

The operators below support the `PrecisionType` parameter.

| Operator| Description|
|:-----|:-----|
| div | Elementwise division|
| fmod | Elementwise modulo|
| remainder | Elementwise remainder|
| pow | Elementwise power operation|
| exp | Exponential operation|
| sqrt | Root extraction operation|
| rsqrt | Root reciprocal operation|
| log | Logarithmic operation|
| log2 | Logarithm operation with 2 as the base|
| log10 | Logarithm operation with 10 as the base|
| reciprocal | Reciprocal operation|

## Examples

```python
import pypto

# Create tensors.
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)

# Use the high-precision mode.
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)

# Use the instruction mode.
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)

# The high-precision mode is used by default.
out = pypto.div(a, b)

# Use an operator (the high-precision mode is automatically used).
out = a / b

# Other operator examples
out = pypto.exp(a, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.sqrt(a, pypto.PrecisionType.INTRINSIC)
out = pypto.log(a, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.pow(a, b, pypto.PrecisionType.HIGH_PRECISION)
```
