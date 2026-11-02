# pypto.log10

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:58:12.095Z pushedAt=2026-10-28T11:57:29.970Z nl=2 -->

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

Performs a base-10 logarithmic operation on **input**.

## Prototype

```python
pypto.log10(input, precision_type=pypto.PrecisionType.INTRINSIC) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| input | Tensor | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Supported dimensions: 1 to 4.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | PrecisionType, optional | Precision mode of the logarithm operation. The default value is `PrecisionType.INTRINSIC`.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss. |

## Return Value

Returns the output tensor, whose data type is the same as that of **input** and whose shape has the same size as **input**.

## Constraints

1. The input tensor and the output tensor must have the same type.
2. Input elements must be greater than 0; otherwise, the output is NaN.

## TileShape Setting Example

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Examples

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log10(x)
```

The results are as follows:

```python
Input data x: [1.0     2.0     3.0]
Output data y: [0.0000 0.3010 0.4771]
```

### High-Precision Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log10(x, pypto.PrecisionType.HIGH_PRECISION)
```

### Instruction Mode Example

```python
x = pypto.tensor([3], pypto.DT_FP16)
y = pypto.log10(x, pypto.PrecisionType.INTRINSIC)
```
