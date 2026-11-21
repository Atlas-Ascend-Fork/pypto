# pypto.ceil_div

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

Divides each element of self by the element at the corresponding position in other and rounds up the result. The computation formula is as follows:

$$
res_i = ceil(self_i \div other_i)
$$

## Prototype

```python
ceil_div(self: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| self  | Input     | Source operand.<br>Supported type: Tensor.<br>The supported data type of the tensor is DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other  | Input    | Source operand.<br>Supported types: Tensor and int.<br>The supported data type of the tensor is DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is the same as that of self and other, and its shape is the broadcast shape of self and other.

## Constraints

1. The self and other types must be the same.
2. Multi-dimensional broadcasting is supported.
3. The value ranges of self and other must be within \[-2^24, 2^24\] to ensure accurate conversion to float32 during computation. **other must not be 0**. When the result of integer division is 0, the result is determined by the chip and may be INT32_MAX or INT32_MIN.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

In a non-broadcast scenario, if the input self shape is [m, n], other is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input self shape is [m, n], other is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.ceil_div(a, b)
```

Result example:

```python
Input data a:    [[2 4 6]]
Input data b:    [[4 2 5]]
Output data out:  [[1 2 2]]
```
