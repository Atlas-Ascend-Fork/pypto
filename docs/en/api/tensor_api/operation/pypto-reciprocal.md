# pypto.reciprocal

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

Computes the element-wise reciprocal of the input tensor, that is, `out = 1 / input`.

## Prototype

```python
pypto.reciprocal(input, precision_type=pypto.PrecisionType.HIGH_PRECISION) -> Tensor
```

## Parameters

| Parameter| Type| Description|
|:-----|:-----|:-----|
| input | Tensor | Input tensor.<br>Supported data types: DT_FP16, DT_BF16, and DT_FP32. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. 1 to 4 dimensions are supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| precision_type | (Optional) PrecisionType| Precision mode of the reciprocal operation. The default value is `PrecisionType.HIGH_PRECISION`.<br>**HIGH_PRECISION**: uses a higher-precision computation method to reduce precision loss.<br>**INTRINSIC**: directly uses chip instructions for computation, which is faster.|

## Return Value

| Type| Description|
|:-----|:-----|
| Tensor | New tensor containing the element-wise reciprocal of the input tensor.|

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

### Example 1: Basic Usage

```python
import pypto

x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x)

# Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
# Output y: [-2.1763, -0.4713, -0.6986,  1.3702]
```

### Example 2: Using the High-Precision Mode

```python
import pypto

# Use the high-precision mode for FP16 computation.
x = pypto.tensor([4], pypto.DT_FP16)
y = pypto.reciprocal(x, pypto.PrecisionType.HIGH_PRECISION)

# Input x:  [4]
# Output y: [0.25]
```

### Example 3: Using the Instruction Mode

```python
import pypto

# Use the instruction mode.
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.reciprocal(x, pypto.PrecisionType.INTRINSIC)

# Input x:  [4]
# Output y: [0.25]
```

## Related APIs

- [pypto.rsqrt](pypto-rsqrt.md): Computes the element-wise reciprocal of the square root of the input tensor.
- [pypto.div](pypto-div.md): Computes the element-wise division of two tensors.
