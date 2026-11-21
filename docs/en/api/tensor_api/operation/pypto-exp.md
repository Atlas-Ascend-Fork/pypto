# pypto.exp

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

Computes the exponential of e for each element in the input tensor element-wise and returns a tensor with the same shape as the input.

## Prototype

```python
pypto.exp(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter| Type| Description|
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_BF16, and DT_FP32. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. 1 to 4 dimensions are supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | (Optional) PrecisionType| Precision mode of the exponential operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: uses a higher-precision computation method to reduce precision loss.|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

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

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp(x)
```

Result example:

```python
Input data x: [0.0    1.0    2.0]
Output data y: [1.0000  2.7183  7.3891]
```

### High-Precision Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.exp(x, pypto.PrecisionType.INTRINSIC)
```
