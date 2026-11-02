# pypto_pro.language.min

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:47:05.669Z pushedAt=2026-10-28T11:57:29.281Z nl=2 -->

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

Obtains the smaller of two scalars.

## Prototype

```python
pypto_pro.language.min(
    lhs: Scalar,
    rhs: Scalar,
    /,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| lhs | Input | Left operand, which is an integer or floating-point constant, or a runtime integer or floating-point scalar expression. |
| rhs | Input | Right operand, which is an integer or floating-point constant, or a runtime integer or floating-point scalar expression. |

## Constraints

- lhs and rhs must be numeric scalars. When both operands are integers or both are floating-point numbers, the result is promoted to the data type with the larger bit width. When one operand is an integer and the other is a floating-point number, both operands are converted to FP32 before comparison.
- The API accepts only two positional arguments and does not support keyword arguments.
- Scope: This API is used only for obtaining the smaller of two scalars. To obtain the element-wise minimum of tiles, use pypto_pro.language.minimum.

## Return Value

Returns the smaller of the two scalars, of the Scalar type. The data type of the return value is determined by the promoted data type of the arguments.

## Examples

### Basic Usage

```python
import pypto_pro.language as pl

# Loop boundary computation (tail block handling).
m_size = pl.min(m_dim - i, 64)
n_size = pl.min(n_dim - j, 128)

# Comparison with integer constants.
causal_kv_tiles = pl.min(qi + 1, skv_tiles)

# Between scalar variables.
c = pl.min(a, b)

# Floating-point scalar.
scale = pl.min(pl.const(2.5, pl.DT_FP32), pl.const(-1.5, pl.DT_FP32))
```
