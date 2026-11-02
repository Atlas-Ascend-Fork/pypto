# pypto.lt

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:03:24.584Z pushedAt=2026-10-28T11:57:29.996Z nl=2 -->

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

Performs an element-wise less-than comparison.

## Prototype

```python
lt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor. The supported Tensor data types vary by model. For details, see [Constraints](#constraints). The data types of the two source operands must be the same.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported types: Tensor, float, and Element.<br>When the type is float, it is automatically converted to Element, and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The supported Tensor and Element data types vary by model. For details, see [Constraints](#constraints).<br>The data types of the two source operands must be consistent.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor of data type `DT\_BOOL` with the same shape as the input tensor. For each position, the output value is `True` if the corresponding element in `input` is strictly less than the corresponding element in `other`; otherwise, the output value is `False`.

## Constraints

1. The types of **input** and **other** must be the same.
2. Supports broadcasting across multiple dimensions to the same shape.
3. Data types of Tensor and Element:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_FP32, DT_INT16, DT_INT64, DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_FP32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_FP32
   <!-- end id6 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through set_vec_tile_shapes.

The TileShape must have the same number of dimensions as the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.lt(a, b)
```

The results are as follows:

```python
Input data a: [1.0 2.0 3.0]
Input data b: [2.0 2.0 2.0]
Output data out: [True, False, False]
```
