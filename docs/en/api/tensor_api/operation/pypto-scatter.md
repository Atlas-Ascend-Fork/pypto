# pypto.scatter

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:29:31.672Z pushedAt=2026-10-28T11:57:30.085Z nl=2 -->

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

Writes the value of src into input. The write position is specified by index. The 3D computation formula is as follows, and other dimensions follow the same pattern:

When src is a fixed scalar:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$$

When src is a tensor:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$$

## Prototype

```python
scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

non-inplace version of scatter\_. For details, see [pypto.scatter\_](pypto-scatter_.md)

## Parameters

For details about the parameters, see [pypto.scatter_](pypto-scatter_.md).

## Return Value

For details about the return value, see [pypto.scatter_](pypto-scatter_.md).

## Constraints

1. For details about the constraints, see [pypto.scatter_](pypto-scatter_.md).
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

For details about the example, see [pypto.scatter_](pypto-scatter_.md).
