# pypto.rsqrt

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:25:54.029Z pushedAt=2026-10-28T11:57:30.074Z nl=2 -->

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

Computes the reciprocal square root of each element in the input tensor, performing element-wise operations. Returns NaN when the input is negative and Inf when the input is zero.

## Prototype

```python
pypto.rsqrt(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of **Tensor**: **DT_FP32**, **DT_FP16**, and **DT_BF16**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | PrecisionType, optional | Precision mode of the reciprocal square root operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss. |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the reciprocal square roots of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n] and the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.rsqrt(x)
```

The results are as follows:

```python
Input data x: [[1.0  4.0], [16.0  9.0]]
Output data y: [[1.0  0.5], [0.25  0.33333]]
```

### High-Precision Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.INTRINSIC)
```
