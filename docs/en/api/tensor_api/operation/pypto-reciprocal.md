# pypto.reciprocal

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:20:26.369Z pushedAt=2026-10-28T11:57:30.053Z nl=2 -->

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

Computes the element-wise reciprocal of the input tensor, that is, `out = 1 / input`.

## Prototype

```python
pypto.reciprocal(input, precision_type=pypto.PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter | Type | Description |
|:-----|:-----|:-----|
| input | Tensor | Input tensor.<br>Supported data types: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. Supported dimensions: 1 to 4. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| precision_type | PrecisionType, optional | Precision mode of the reciprocal operation. The default value is `PrecisionType.HIGH_PRECISION`.<br>**HIGH_PRECISION**: Uses a higher-precision computation method to reduce precision loss.<br>**INTRINSIC**: Directly uses chip instructions for computation, which is faster. |

## Return Value

| Type | Description |
|:-----|:-----|
| Tensor | New tensor containing the element-wise reciprocal of the input tensor. |

## Constraints

1. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### Example 1: Basic Usage

```python
import pypto

x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x)

# Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
# Output y: [-2.1763, -0.4713, -0.6986,  1.3702]
```

### Example 2: Using High-Precision Mode

```python
import pypto

# Use high-precision mode for FP16 computation.
x = pypto.tensor([4], pypto.DT_FP16)
y = pypto.reciprocal(x, pypto.PrecisionType.HIGH_PRECISION)

# Input x:  [4]
# Output y: [0.25]
```

### Example 3: Using the instruction mode

```python
import pypto

# Use instruction mode.
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x, pypto.PrecisionType.INTRINSIC)

# Input x:  [4]
# Output y: [0.25]
```

## Related APIs

- [pypto.rsqrt](pypto-rsqrt.md): Computes the element-wise reciprocal of the square root of the input tensor.
- [pypto.div](pypto-div.md): Computes the element-wise division of two tensors.
