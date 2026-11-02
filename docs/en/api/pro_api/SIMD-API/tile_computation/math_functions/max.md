# pypto_pro.language.max

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:46:32.954Z pushedAt=2026-10-28T11:57:29.278Z nl=2 -->

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

Returns the larger of two scalars.

## Prototype

```python
pypto_pro.language.max(
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

- lhs and rhs must be numeric scalars. When both operands are integers or both are floating-point, the result is promoted to the data type with the larger bit width. When one operand is an integer and the other is floating-point, both operands are converted to FP32 before comparison.
- The API accepts only two positional arguments and does not support keyword arguments.
- Scope: This API is used only for obtaining the larger of two scalars. For element-wise maximum on tiles, use [pypto_pro.language.maximum](../elementwise/maximum.md).

## Return Value

Returns the larger of the two scalars, of the Scalar type. The data type of the return value is determined by the promotion result of the argument data types.

## Examples

### Basic Usage

```python
import pypto_pro.language as pl

# Ensure the minimum value.
bottom_k = pl.max(1, trunk_len - topk + 1)

# Loop boundary.
end = pl.max(start, min_end)

# Between scalar variables.
c = pl.max(a, b)

# Floating-point scalar.
scale = pl.max(pl.const(2.5, pl.DT_FP32), pl.const(-1.5, pl.DT_FP32))
```
