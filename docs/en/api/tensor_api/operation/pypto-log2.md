# pypto.log2

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

Performs a base-2 logarithmic operation on input.

## Prototype

```python
pypto.log2(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter| Type| Description|
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Supported dimensions: 1 to 4.<br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | (Optional) PrecisionType| Precision mode of the logarithmic operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: uses a higher-precision computation method to reduce precision loss.|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. The input and output tensors must have the same type.
2. Input elements must be greater than 0; otherwise, the output is NaN.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## TileShape Setting Example

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Examples

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log2(x)
```

Result example:

```python
Input data x: [1.0     2.0     3.0]
Output data y: [0.0000 1.0000 1.5849]
```

### High-Precision Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log2(x, pypto.PrecisionType.INTRINSIC)
```
