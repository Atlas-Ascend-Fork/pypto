# pypto.expm1

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Computes the natural exponential of each element of the input tensor and then subtracts 1. The formula is as follows:
$$
y_i = e^{x_i} - 1
$$

## Prototype

```python
expm1(input: Tensor) -> Tensor
```

## Parameters

| Parameter     | Input/Output| Description                                                                                                                                                            |
|----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor. When the input data type is DT_FP32, DT_FP16, or DT_BF16, its data type is the same as that of the input tensor. When the input data type is DT_INT32 or DT_INT16, its data type is DT_FP32. Each element is the natural exponential of the corresponding element of the input tensor minus 1.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.expm1(x)
```

Result example:

```python
Input data x: [[1., 2.], [3., 4.]]
Output data y: [[1.7183, 6.3891], [19.0855, 53.5981]]
```
