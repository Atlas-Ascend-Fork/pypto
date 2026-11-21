# pypto.concat

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id10 -->
- Kirin 9050 processors: Supported
<!-- end id10 -->

## Description

Concatenates multiple input tensors along the specified dimension (dim) and returns a concatenated tensor.

## Prototype

```python
concat(tensors: List[Tensor], dim: int = 0) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                                                                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tensors | Input     | Source operand. Supported type: Tensor. A tensor supports the following data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_FP32, DT_BF16, DT_INT64, and DT_UINT64. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints). <!-- end id9 -->Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension along which concatenation is performed. Supported data type: int. The default value is 0.                                                                                                                                                     |

## Return Value

Returns the output tensor. Its data type is the same as that of any tensor defined by tensors, and its shape is the same as that of any tensor defined by tensors (except for the dimension corresponding to dim). The dimension corresponding to dim is the sum of the corresponding dimensions of all tensors defined by tensors.

## Constraints

1. The size of the source operand tensors must be at least 2, that is, len\(tensors \)\>=2, and no more than 128. (Input of a single tensor is supported, but its precision is not guaranteed.)

2. The input tensors must have the same data type, the same number of dimensions, and the same value for each dimension except the dimension (dim) to be concatenated. For the dimension to be concatenated, validShape must equal the corresponding dimension value of the tensor, and for the remaining dimensions, validShape of all tensors must be the same.

3. dim: -input.dim <= dim < input.dim (input corresponds to any tensor defined by tensors).

4. When setting viewshape, do not tile the dimension corresponding to dim (this means the value of viewshape is greater than or equal to that of any tensor defined by tensors).

5. The valid shape of the output tensor must be correctly specified prior to calling concat, as this API does not perform automatic inference.

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

<!-- npu="x90,9030,9050" id6 -->
7. A tensor supports the following data types:
   <!-- npu="9030" id7 -->
   - Kirin 9030 processors: DT_INT32, DT_INT16, DT_INT8, DT_FP16, and DT_FP32
   <!-- end id7 -->
   <!-- npu="x90" id8 -->
   - Kirin X90 processors: DT_INT32, DT_INT16, DT_INT8, DT_FP16, and DT_FP32
   <!-- end id8 -->
   <!-- npu="9050" id11 -->
   - Kirin 9050 processors: DT_INT32, DT_INT16, DT_INT8, DT_FP16, and DT_FP32
   <!-- end id11 -->
<!-- end id6 -->

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input tensor dimensions are [m, c1, p] and [m, c2, p], the output is [m, c1+c2, p], and TileShape is set to [m1, n1, p1], then m1 and p1 are used to tile the m and p axes, respectively, and n1 is used to tile the c1 and c2 axes.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
a = pypto.ones([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 1s
b = pypto.zeros([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 0s
out = pypto.concat([a, b], dim = 0)
```

Result example:

```python
Input data a:   [[1.0 1.0],
              [1.0 1.0]]
Input data b:   [[0.0 0.0],
              [0.0 0.0]]
Output data out: [[1.0 1.0],
              [1.0 1.0],
              [0.0 0.0],
              [0.0 0.0]]

```
