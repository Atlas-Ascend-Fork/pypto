# pypto.gcd

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

Computes the greatest common divisor of the elements of input and other.

## Prototype

```python
gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                                                                                                                                                                                                  |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).          |
| other  | Input     | Source operand.<br>Supported types: int and Tensor.<br>Supported data types of a tensor: DT_INT8, DT_INT16, DT_INT32, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is the same as that of input and other, and its shape is the broadcast shape of input and other.

## Constraints

1. The data types of input and other must be the same.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes. The last axis of TileShape must be 32-byte aligned.

TileShape must have the same dimensions as the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], other is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], other is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)
y = pypto.tensor([2, 3], pypto.DT_INT32)
z = pypto.gcd(x, y)
# Using a scalar
c = pypto.gcd(x, 2)
```

Result example:

```python
Input data x: : [[9 9 9],
             [6 6 6]]
Input data y:   [[1, 2, 3],
             [1 2 3]]
Output data z:   [[1, 1, 3],
             [1 2 3]]
Output data c:   [[1, 1, 1],
             [2 2 2]]
```
