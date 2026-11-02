# PrecisionType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:47:53.742Z pushedAt=2026-10-28T11:57:29.680Z nl=2 -->

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

**PrecisionType** defines the precision mode of high-precision vector operators, which is used to control how precision is handled in computations such as division, modulo, exponential, and logarithm.

## Prototype

```python
class PrecisionType(enum.Enum):
    INTRINSIC = ...       # Instruction mode. Directly uses chip instructions.
    HIGH_PRECISION = ...  # High-precision mode.
```

## Parameters

| Parameter Value | Description |
|:-------|:-----|
| HIGH_PRECISION | High-precision mode. Uses higher-precision computation in the underlying implementation. |
| INTRINSIC | Instruction mode. Directly uses chip instructions for computation, offering higher performance. |

## Usage Recommendations

1. **Default behavior**: If no precision mode is specified, `HIGH_PRECISION` mode is used by default to ensure computation precision.
2. **Scenarios with high precision requirements**: `HIGH_PRECISION` mode is recommended, as it can effectively reduce precision loss and improve the accuracy of computation results.
3. **Scenarios with low precision requirements but a focus on performance**: `INTRINSIC` mode can be used to perform computations directly using chip instructions.
4. HIGH_PRECISION usage notes:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: Supported
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: Not supported
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: Not supported
   <!-- end id6 -->

## Supported Operators

The following operators support the `PrecisionType` parameter:

| Operator | Description |
|:-----|:-----|
| div | Element-wise division |
| fmod | Element-wise modulo |
| remainder | Element-wise remainder |
| pow | Element-wise power |
| exp | Exponential operation |
| sqrt | Square root |
| rsqrt | Reciprocal square root |
| log | Logarithm |
| log2 | Base-2 logarithm |
| log10 | Base-10 logarithm |
| reciprocal | Reciprocal |

## Example

```python
import pypto

# Create a tensor.
a = pypto.tensor([1, 3], pypto.DT_FP16)
b = pypto.tensor([1, 3], pypto.DT_FP16)

# Use the high-precision mode.
out = pypto.div(a, b, pypto.PrecisionType.HIGH_PRECISION)

# Use the instruction mode.
out = pypto.div(a, b, pypto.PrecisionType.INTRINSIC)

# Use the high-precision mode by default.
out = pypto.div(a, b)

# Use operators (automatically using the high-precision mode).
out = a / b

# Examples of other operators.
out = pypto.exp(a, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.sqrt(a, pypto.PrecisionType.INTRINSIC)
out = pypto.log(a, pypto.LogBaseType.LOG_E, pypto.PrecisionType.HIGH_PRECISION)
out = pypto.pow(a, b, pypto.PrecisionType.HIGH_PRECISION)
```
