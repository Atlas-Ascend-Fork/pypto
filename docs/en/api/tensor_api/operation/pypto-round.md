# pypto.round

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->

## Description

Rounds the elements of the input tensor to the specified number of decimal places. If the value is equally distant from the two decimals at the specified position, the even number at the specified position is taken.

## Prototype

```python
round(input: Tensor, decimals: int = 0) -> Tensor
```

## Parameters

| Parameter     | Input/Output| Description                                                                                                                                                            |
|----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | Input     | Source operand.<br>Supported type: Tensor.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| decimals | Input     | Number of decimal places to round to.<br>Type: int.                                                                                                                                 |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the result of rounding the corresponding elements of the input tensor to the specified number of decimal places.

## Constraints

1. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, and DT_INT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, and DT_INT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, and DT_INT32
   <!-- end id6 -->
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
3. When the data type of the input tensor is DT_INT32 or DT_INT64, decimals must be 0. In this case, the returned result is the same as the input tensor.

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
y = pypto.round(x, decimals=1)
```

Result example:

```python
Input data x: [[1.21, 2.35], [3.65, 4.76]]
Output data y: [[1.2, 2.4], [3.6, 4.8]]
```
