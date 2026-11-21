# pypto_pro.language.min

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

In the kernel code, you can directly use the built-in Python function min(lhs, rhs). During compilation, the function is automatically converted to pypto_pro.language.min(lhs, rhs), which returns the smaller value of the two scalars.

## Prototype

```python
pypto_pro.language.min(
    lhs: Union[int, float],
    rhs: Union[int, float],
    /,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Left operand (a Python int, Python float, or integer or floating-point scalar expression in the kernel).|
| rhs | Input| Right operand (a Python int, Python float, or integer or floating-point scalar expression in the kernel).|

## Constraints

- **Scalars only**: Used in scenarios such as loop boundaries and index computation.
- **Elementwise minimum value of tiles**: pypto_pro.language.minimum is used.
- **No multi-parameter support**: Only two parameters are accepted. min(a, b, c) is not supported.
- **Same-type constraint**: Both operands must be of the same type (either integers or floating-point numbers). Mixing the two types ( int or float) will result in an error.

## Return Value

Returns the smaller scalar value between lhs and rhs.

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
