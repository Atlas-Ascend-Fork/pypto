# pypto.rsqrt

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

Computes the reciprocal of the square root of each element in the input tensor, performing element-wise operations. Returns NaN when the input is a negative number and Inf when the input is zero.

## Prototype

```python
pypto.rsqrt(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter| Type| Description|
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. 1 to 4 dimensions are supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | (Optional) PrecisionType| Precision mode of the reciprocal of the square root operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: uses a higher-precision computation method to reduce precision loss.|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the reciprocal of the square root of the corresponding elements of the input tensor.

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
<!-- npu="x90,9030,9050" id6 -->
2. A tensor supports the following data types:
   <!-- npu="9030" id7 -->
   - Kirin 9030 processors: DT_FP16 and DT_FP32
   <!-- end id7 -->
   <!-- npu="x90" id8 -->
   - Kirin X90 processors: DT_FP16 and DT_FP32
   <!-- end id8 -->
   <!-- npu="9050" id11 -->
   - Kirin 9050 processors: DT_FP16 and DT_FP32
   <!-- end id11 -->
<!-- end id6 -->

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
y = pypto.rsqrt(x)
```

Result example:

```python
Input data x: [[1.0 4.0], [16.0 9.0]]
Output data y: [[1.0 0.5], [0.25 0.33333]]
```

### High-Precision Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP16)
y = pypto.rsqrt(x, pypto.PrecisionType.INTRINSIC)
```
