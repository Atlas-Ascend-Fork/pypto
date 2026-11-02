# pypto_pro.language.max

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:18:24.288Z pushedAt=2026-10-28T11:57:29.518Z nl=2 -->

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

In Kernel code, you can directly use the Python built-in max(lhs, rhs). At compile time, it is automatically converted to pypto_pro.language.max(lhs, rhs), which returns the larger of two scalars.

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
| lhs | Input | Left operand (a Python int, Python float, or an integer or floating-point scalar expression in the Kernel). |
| rhs | Input | Right operand (a Python int, Python float, or an integer or floating-point scalar expression in the Kernel). |

## Constraints

- **Scalar only**: Used for loop bounds, index computation, and similar scenarios.
- **Element-wise maximum for tiles**: Use [pypto_pro.language.maximum](../../SIMD-API/tile_computation/elementwise/maximum.md).
- **No multi-parameter support**: Accepts exactly two parameters; max(a, b, c) is not supported.
- **Same-type constraint**: Both operands must be either integers or floating-point numbers; mixing int and float causes an error.

## Return Value

Returns the larger scalar value between lhs and rhs.

## Examples

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    bottom_k = max(1, trunk_len - topk + 1)
    end = pl.max(start, min_end)

    offset = max(i - window, 0)
    c = max(a, b)
```
