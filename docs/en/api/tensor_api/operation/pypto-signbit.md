# pypto.signbit

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:32:24.075Z pushedAt=2026-10-28T11:57:30.095Z nl=2 -->

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

Checks whether the sign bit of each element in the input tensor is set (that is, whether the element is negative). This is an element-wise operation.

Logic:

- If the element is negative (including −∞ and −0.0), **True** is returned.
- If the element is positive (including +∞ and +0.0) or NaN, **False** is returned.

## Prototype

```python
signbit(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of **Tensor**: **DT_FP16**, **DT_BF16**, **DT_FP32**, **DT_INT8**, **DT_INT16**, and **DT_INT32**.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor whose shape is the same as that of the input tensor, with the data type **DT_BOOL**. Each element indicates whether the sign bit of the corresponding element in the input tensor is set (True for negative numbers and False for non-negative numbers).

## Constraints

1. The TileShape must have the same dimension count as the input.
2. Due to temporary memory usage, when the input data type is **DT_FP32**, the **TileShape** size has an additional constraint. Assuming **TileShape** is [a,b,c,d], then a*b*c*d*sizeof(self) + a*b*c*d*sizeof(FP16) + a*b*c*d*sizeof(UINT8) < UB.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([-5, 0, 5, 10, -2], pypto.DT_FP32)
y = pypto.signbit(x)
```

The results are as follows:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [True, False, False, False, True]
```
