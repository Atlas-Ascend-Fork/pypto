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

Returns the smaller value between two scalars.

## Prototype

```python
pypto_pro.language.min(lhs: Union[int, float], rhs: Union[int, float], /) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| lhs | Input| Left operand, which is an integer or floating-point constant, or a runtime integer or floating-point scalar expression.|
| rhs | Input| Right operand, which is an integer or floating-point constant, or a runtime integer or floating-point scalar expression.|

## Return Value

Returns the smaller value between two scalars. The return value is of the Scalar type. The return value's data type is determined by the result of data type promotion among the parameters.

## Constraints

- lhs and rhs must be numeric scalars. If both operands are integers or both are floating-point values, the result is promoted to the data type with the larger bit width. If one operand is an integer and the other is floating-point, both are converted to FP32 before comparison.
- This API accepts only two positional arguments and does not support keyword arguments.
- Applicable scope: This API is only for scalar minimum. For element-wise minimum on Tiles, use pypto_pro.language.minimum.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scalar_min_kernel(
    lhs: pl.DT_FP32,
    rhs: pl.DT_FP32,
    out: pl.Tensor[[1], pl.DT_FP32],
):
    value = pl.min(lhs, rhs)
    pl.setval(out, 0, value)
```

For example, when lhs=2.5 and rhs=–1.5, out[0] is 2.5. Integer scalars can be used in a Kernel in the same way for loop bound or tail block size computation.
