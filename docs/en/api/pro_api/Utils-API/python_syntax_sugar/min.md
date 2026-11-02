# pypto_pro.language.min

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:19:08.419Z pushedAt=2026-10-28T11:57:29.525Z nl=2 -->

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

In Kernel code, you can directly use the Python built-in min(lhs, rhs). At compile time, it is automatically converted to pypto_pro.language.min(lhs, rhs), which returns the smaller of two scalars.

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
| lhs | Input | Left operand (Python int, Python float, or an integer or floating-point scalar expression in Kernel). |
| rhs | Input | Right operand (Python int, Python float, or an integer or floating-point scalar expression in Kernel). |

## Constraints

- **Scalars only**: Used in scenarios such as loop bounds and index computation.
- **Element-wise minimum for Tile**: Use pypto_pro.language.minimum.
- **No multiple arguments**: Accepts exactly two arguments; min(a, b, c) is not supported.
- **Same-type constraint**: Both operands must be either integers or floating-point numbers; mixing int and float causes an error.

## Return Value

Returns the smaller scalar value of lhs and rhs.

## Examples

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    m_size = min(m_dim - i, 64)
    n_size = pl.min(n_dim - j, 128)

    causal_kv_tiles = min(qi + 1, skv_tiles)
    bottom = min(a, 0)
```
