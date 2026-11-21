# pypto.signbit

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

Checks whether the sign bit of each element in the input tensor is set (that is, whether the element is negative), performing element-wise operations.

Logic:

- If the element is a negative number (including −∞ and −0.0), True is returned.
- If the element is a positive number (including +∞ and +0.0) or NaN, False is returned.

## Prototype

```python
signbit(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor, and its data type is DT_BOOL. Its elements indicate whether the sign bit of the corresponding element of the input tensor is set (True for negative numbers and False for non-negative numbers).

## Constraints

1. TileShape must have the same dimensions as those of input.
2. Due to the use of temporary memory, when the input data type is DT_FP32, the size of TileShape has additional constraints. If TileShape is [a,b,c,d], then a*b*c*d*sizeof(self) + a*b*c*d*sizeof(FP16) + a*b*c*d*sizeof(UINT8) < UB.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example: If the input shape is [n], the output is [n], and TileShape is set to [n1], then n1 is used to tile the n axis.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.signbit(x)
```

When the caller provides the following input data, the result is as follows:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [True, False, False, False, True]
```
