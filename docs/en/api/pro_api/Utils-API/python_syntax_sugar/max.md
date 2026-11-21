# pypto_pro.language.max

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

In the kernel code, you can directly use the built-in Python function max(lhs, rhs). During compilation, the function is automatically converted to pypto_pro.language.max(lhs, rhs), which returns the larger value of the two scalars.

## Prototype

```python
pypto_pro.language.max(
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

- **Scalars only**: used in scenarios such as loop boundaries and index computation.
- **Elementwise maximum value of tiles**: [pypto_pro.language.maximum](../../SIMD-API/tile_computation/elementwise/maximum.md) is used.
- **No multi-parameter support**: Only two parameters are accepted. max(a, b, c) is not supported.
- **Same-type constraint**: Both operands must be of the same type (either integers or floating-point numbers). Mixing the two types ( int or float) will result in an error.

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
