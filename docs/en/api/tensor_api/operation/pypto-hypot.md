# pypto.hypot

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

Computes the square root of the sum of squares of input and other (that is, the length of the hypotenuse of a right triangle) element-wise. The computation formula is as follows:

$$
res_i = \sqrt{input_i^2 + other_i^2}
$$

## Prototype

```python
hypot(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is the same as that of input and other, and its shape is the broadcast shape of input and other.

## Constraints

1. If both input and other are tensors, their data types must be the same.
2. For the BF16 and FP16 types, the internal computation may increase precision to avoid intermediate overflow.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example: If the shapes of both input and other are [n], the output is [n], and TileShape is set to [n1], then n1 is used to tile the n axis.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

```python
# Example: Compute the hypotenuse of two sets of right-angle sides.
# First set: (3, 4) -> 5
# Second set: (5, 12) -> 13
a = pypto.tensor([2], pypto.DT_FP32)
b = pypto.tensor([2], pypto.DT_FP32)
out = pypto.hypot(a, b)
```

When the caller provides the following input data, the result is as follows:

```python
Input data a:   [3.0, 5.0]
Input data b:   [4.0, 12.0]
Output data out: [5.0, 13.0]
```
