# pypto.triu

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

Returns the upper triangular part of a 2D tensor or a batch of tensors. Other elements of the resulting tensor are set to 0.

## Prototype

```python
triu(input: Tensor, diagonal: SymInt = 0) -> Tensor
```

## Parameters

| Parameter  | Input/Output| Description                                                                                                                                                                                                                       |
| -------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input    | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 2 to 5 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| diagonal | Input     | Diagonal offset. The default value is 0 (main diagonal).<br>Type: SymInt.                                                                                                                                                    |

## Return Value

Returns a tensor whose shape and data type are the same as those of the input.

## Constraints

1. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_UINT16, DT_UINT32, DT_INT64, and DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, and DT_INT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_INT16, and DT_INT32
   <!-- end id6 -->

2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([3, 3], pypto.DT_INT32)        # shape (3, 3)
diagonal = 0
out = pypto.triu(x, diagonal)
```

Result example:

```python
Input data x: [[1 2 3],
             [4 5 6],
             [7 8 9]]
Output data out: [[1 2 3],
             [0 5 6],
             [0 0 9]]                             # shape (3, 3)
```
