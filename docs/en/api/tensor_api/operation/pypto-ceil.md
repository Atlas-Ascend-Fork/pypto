# pypto.ceil

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:14:11.196Z pushedAt=2026-10-28T11:57:29.808Z nl=2 -->

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

Computes the ceiling of each element in the input tensor (returns the smallest integer not less than the element) in an element-wise manner. Integer values are returned as-is, while floating-point values are rounded up.

## Prototype

```python
ceil(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                          |
|-----------------|--------------|----------------------------------------------------------------------|
| input           | Input        | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16. The shape supports only 1 to 4 dimensions. |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the ceiling values of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.ceil(x)
```

The results are as follows:

```python
Input data x: [1.2, 4.7, -1.1, 9.0, 3.9]
Output data y: [2.0, 5.0, -1.0, 9.0, 4.0]
```
