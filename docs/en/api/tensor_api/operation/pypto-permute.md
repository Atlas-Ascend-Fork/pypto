# pypto.permute

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:14:13.541Z pushedAt=2026-10-28T11:57:30.034Z nl=2 -->

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

Returns a tensor that is the transposed version of the input tensor. The dimensions of the input tensor are rearranged according to the specified dimension order. This operator does not change the total number or content of elements in the tensor; it only changes the arrangement of the dimensions.

## Prototype

```python
permute(input: Tensor, perm: list[int]) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 5 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| perm    | Input      | Dimension order list. It must be a permutation containing all dimension indices, with a length equal to the number of dimensions of the input tensor. Each dimension index ranges from 0 to ShapeSize-1 and must not be repeated. |

## Return Value

Returns a tensor with the same data type as the input, whose dimension order is rearranged according to the order specified by **perm**.

## Constraints

1. The input tensor and the output tensor must have the same data type.
2. 64-bit integer format restriction: **DT_INT64** and **DT_UINT64** do not support the NZ (Fractal-Z) format and support only the ND format.
3. 32-byte alignment constraint on the last axis of TileShape (only for scenarios where no actual permutation is required): When **permute** determines that no actual permutation is required and directly returns the input tensor (that is, the input is a 1-dimensional tensor, or **perm** is an identity permutation), the number of bytes of the last dimension of TileShape must be aligned to 32 bytes (**BLOCK_SIZE**). That is: `TileShape last dimension element count × sizeof(data type) % 32 == 0`.

   Note: In this scenario, the input Tensor is directly returned (`return self`), but **TILE_REGISTER_COPY** in the subsequent flow still validates the 32-byte alignment of the last axis of the TileShape. If this constraint is not satisfied, the following error is reported: `CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0`.

4. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FP8E4M3, DT_FP8E5M2, DT_HF8, DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL
   <!-- end id6 -->
5. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**. The dimensions of the TileShape must be consistent with the input **input**.

Example: If the input **input** shape is [2, 3, 4] and the target permutation is [2, 0, 1], the TileShape can be set to [2, 3, 4] or to an appropriate value based on the splitting requirements.

```python
pypto.set_vec_tile_shapes(2, 3, 4)
```

### API Call Example

```python
x = pypto.tensor([1, 2, 3, 4], pypto.DT_FP32)
perm = [3, 1, 0, 2]
y = pypto.permute(x, perm)
```

The results are as follows:

```python
Input data x: [[[[ 0.9586, -0.4325,  0.7582, -2.6209],
              [ 1.0931, -0.3324, -2.3653, -0.0324],
              [ 1.6083,  1.3619, -0.1481,  0.4394]],

             [[ 0.2353, -0.7177, -0.4954,  0.4158],
              [-0.9788, -1.4224,  0.2558,  1.5322],
              [-0.6645,  2.1023,  0.8968,  0.8690]]]],

Output data y: [[[[ 0.9586,  1.0931,  1.6083]],
             [[ 0.2353, -0.9788, -0.6645]]],

            [[[-0.4325, -0.3324,  1.3619]],
             [[-0.7177, -1.4224,  2.1023]]],

            [[[ 0.7582, -2.3653, -0.1481]],
             [[-0.4954,  0.2558,  0.8968]]],

            [[[-2.6209, -0.0324,  0.4394]],
             [[ 0.4158,  1.5322,  0.8690]]]]
```
