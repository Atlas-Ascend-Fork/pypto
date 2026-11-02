# pypto.maximum

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:04:54.837Z pushedAt=2026-10-28T11:57:29.999Z nl=2 -->

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

Computes the element-wise maximum of the input and another input. Supports 2D, 3D, or 4D tensors.

## Precautions

- **SymbolicScalar parameters are not supported**: To compare **SymbolicScalar** values, use the [SymbolicScalar.max()](../symbolic/pypto-SymbolicScalar-max.md) method.
- At least one of the two parameters must be of the Tensor type.

## Prototype

```python
maximum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported types are int, float, Element, and Tensor.<br>When the type is int or float, it is automatically converted to the Element type, where int corresponds to DT_INT32 and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| other   | Input      | Source operand.<br>Supported types are int, float, Element, and Tensor.<br>When the type is int or float, it is automatically converted to the Element type, where int corresponds to DT_INT32 and float corresponds to DT_FP32. To use other data types, construct them through Element.<br>The Tensor and Element data types supported vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>The type and data type must be the same as those of the source operand. |

At least one of Source Operand 1 and Source Operand 2 must be a tensor.

## Return Value

When both source operands are tensors, the two tensors must satisfy the broadcast relationship. This API returns a tensor with the same shape as the broadcast result of Source Operand 1 and Source Operand 2, with the same data type as the source operands, whose elements are the element-wise maximum of Source Operand 1 and Source Operand 2. When the source operands are tensors, both Source Operand 1 and Source Operand 2 support multi-axis broadcasting.

When one of the two source operands is a tensor, a tensor with the same shape as the input tensor is returned, whose elements are the element-wise maximum of Source Operand 1 and Source Operand 2.

## Constraints

1. When both inputs are of the Tensor type, the supported data types are as follows:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_UINT32, DT_FP32, DT_INT16, DT_UINT16, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_INT64, DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id6 -->
2. When one input is of the Tensor type and the other is of the Element type, the supported data types are as follows:
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16, DT_INT64, DT_UINT64
   <!-- end id7 -->
   <!-- npu="A3" id8 -->
   - Atlas A3 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id8 -->
   <!-- npu="910b" id9 -->
   - Atlas A2 products: DT_INT32, DT_INT16, DT_FP16, DT_FP32, DT_BF16
   <!-- end id9 -->
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([3], pypto.DT_INT32)
b = pypto.tensor([3], pypto.DT_INT32)
out = pypto.maximum(a, b)
```

The results are as follows:

```python
Input data a: [0, 2, 4]
Input data b: [3, 1, 3]
Output data out: [3, 2, 4]
```
