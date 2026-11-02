# pypto.atan2

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:04:50.402Z pushedAt=2026-10-28T11:57:29.767Z nl=2 -->

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

Computes the arctangent value of y/x element-wise.

## Prototype

```python
atan2(y: Tensor, x: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| y       | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| x       | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the arctangent values of the corresponding elements of the input tensor.

## Constraints

1. The data types of y and x must be the same.
2. Due to temporary memory usage, the TileShape size must satisfy the following: if the TileShape is \[a,b,c,d\], then 7\*a\*b\*c\*d\*sizeof\(DT_FP32\) < UB.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
y = pypto.tensor([4], pypto.DT_FP32)
x = pypto.tensor([4], pypto.DT_FP32)
z = pypto.atan2(y, x)
```

The results are as follows:

```python
Input data y: [1.0    -1.0     1.0    -1.0   ]
Input data x: [1.0     1.0    -1.0    -1.0   ]
Output data z: [0.7854  2.3562 -2.3562 -0.7854]
```
