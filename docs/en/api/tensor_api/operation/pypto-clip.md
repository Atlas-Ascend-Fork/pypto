# pypto.clip

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:15:06.627Z pushedAt=2026-10-28T11:57:29.817Z nl=2 -->

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

Clips the input tensor to the specified minimum and maximum value range. Elements smaller than the minimum are replaced with the minimum, elements greater than the maximum are replaced with the maximum, and all other values remain unchanged. This API is non-in-place; it does not modify the input tensor, but returns a new tensor as the output.

## Prototype

```python
clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
)-> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|----------------|--------------|------------------------------------------------------------------------------|
| input  | Input      | Source Operand.<br>Supported type: Tensor.<br>The data types supported by Tensor vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The data dimension supports only 1 to 4 dimensions, and the element count must not exceed INT32_MAX. |
| min    | Input      | Source Operand.<br>Supported types: int\float\Element and Tensor. The data type must be the same as that of the input.<br>When the value is of int or float type, it is automatically converted to the Element type, and the data type is the same as that of the input. To use other data types, construct the value through Element.<br>The data types supported by Tensor and Element vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The data dimension supports only 1 to 4 dimensions, and the element count must not exceed INT32_MAX.<br>Optional. The default value is -INF.<br>NaN, INF, and -INF are defined only for floating-point operations, that is, they take effect only when the data type is DT_FP16/DT_FP32. When the data type is DT_INT16 or DT_INT32, the comparison logic for the default value is skipped. |
| max    | Input      | Source Operand.<br>Supported types: int\float\Element and Tensor. The data type must be the same as that of the input.<br>When the value is of int or float type, it is automatically converted to the Element type, and the data type is the same as that of the input. To use other data types, construct the value through Element.<br>The data types supported by Tensor and Element vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The data dimension supports only 1 to 4 dimensions, and the element count must not exceed INT32_MAX.<br>Optional. The default value is INF.<br>NaN, INF, and -INF are defined only for floating-point operations, that is, they take effect only when the data type is DT_FP16/DT_FP32. When the data type is DT_INT16 or DT_INT32, the comparison logic for the default value is skipped. |

## Return Value

When the input is a scalar, the output is:

$$
Y_{i} = \text{MIN}\left( \text{MAX}\left(X_{i}, \text{min\_value}\right), \text{max\_value} \right)
$$

When the input is a tensor, the output is:

$$
Y_{i} = MIN\left( MAX(X_{i}, min\_value_{i}), max\_value_{i} \right)
$$

The data type of the output tensor is the same as that of the input.

When either min or max is NAN, the output result is NAN.

When min \> max, all elements in the output tensor are set to max.

## Constraints

1. **min** and **max** must have the same type, both being Element or both being Tensor.
2. When **min**/**max** are of the Tensor type, their shape sizes must be broadcastable to the input shape.
3. **min** and **max** support being omitted simultaneously, in which case the original value is returned.
4. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT64.
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16.
   <!-- end id6 -->
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **max** and **min** are [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **max** and **min** are [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = pypto.tensor([2,3], pypto.DT_INT32)
max = pypto.tensor([2,3], pypto.DT_INT32)
out = pypto.clip(x,min,max)
```

The results are as follows:

```python
Input data self: [[-2 1 2], [3 4 5]]
Input data min: [[-1 0 2], [0 3 5]]
Input data max: [[1 2  1], [4 4 4]]
Output data out: [[-1 1 1], [3 4 4]]
```

Example 2:

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = 1
max = 3
out = pypto.clip(x,min,max)
```

The results are as follows:

```python
Input data x: [[0 2 4], [3 4 6]]
Output data out: [[1 2 3], [3 3 3]]
```
