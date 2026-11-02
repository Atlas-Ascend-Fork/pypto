# pypto.exp2

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:29:10.205Z pushedAt=2026-10-28T11:57:29.867Z nl=2 -->

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

Computes the base-2 exponential of each element in the input tensor element-wise, and returns a tensor with the same shape as the input.

## Prototype

```python
exp2(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, and DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns an output tensor. When the input is of type DT_FP32, DT_FP16, or DT_BF16, the output tensor has the same data type and shape as the input. When the input is of type DT_INT32, DT_INT16, DT_INT8, or DT_UINT8, the output tensor is of type DT_FP32 and has the same shape as the input.

## Constraints

1. The value range of **input** must be within \[-2^24, 2^24\] to ensure accurate conversion to float32 during computation.
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
3. Temporary space usage: If TileShape is \[a,b,c,d\], the required temporary space size is
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: a\*b\*c\*d\*sizeof\(DT_FP32\)
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: a\*b\*c\*\(2\*d+3\*d/8\)\*sizeof\(DT_FP32\)
   <!-- end id6 -->

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n] and the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp2(x)
```

The results are as follows:

```python
Input data x: [0.0    1.0    2.0]
Output data y: [1.0    2.0    4.0]
```
