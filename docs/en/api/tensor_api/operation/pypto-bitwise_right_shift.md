# pypto.bitwise_right_shift

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:10:41.409Z pushedAt=2026-10-28T11:57:29.794Z nl=2 -->

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

Shifts **input** right by **other** bits element-wise. The computation formula is as follows:

$$
res_i = input_i >> other_i
$$

## Prototype

```python
bitwise_right_shift(input: Union[Tensor, int], other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                      |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input  | Input      | Source operand.<br>Supported types: **int** and **Tensor**. Supported **Tensor** data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| other  | Input      | Source operand.<br>Supported types: **int** and **Tensor**. Supported **Tensor** data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed **2147483647** (**INT32_MAX**).     |

## Return Value

Returns the output tensor. Its data type is the same as that of **input** and **other**, and its shape is the size after **input** and **other** are broadcast.

## Constraints

1. The types of **input** and **other** must be the same.
2. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT8, DT_UINT8, DT_INT64, DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT16, DT_UINT16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT16, DT_UINT16
   <!-- end id6 -->
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_INT16)
b = pypto.tensor([1, 3], pypto.DT_INT16)
out = pypto.bitwise_right_shift(a, b)
```

The results are as follows:

```python
Input a:    [[1 2 3]]
Input b:    [[1 1 1]]
Output out: [[0 1 1]]
```
