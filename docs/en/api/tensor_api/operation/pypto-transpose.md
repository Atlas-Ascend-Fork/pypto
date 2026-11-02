# pypto.transpose

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:41:35.380Z pushedAt=2026-10-28T11:57:30.132Z nl=2 -->

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

Returns a tensor that is the transposed version of the input tensor. The specified dimensions **dim0** and **dim1** are swapped.

## Prototype

```python
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source Operand.<br>Supported type: Tensor. Supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 5 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**).<br>The operator's support for different shapes varies depending on the specific hardware model. For details, see [Constraints](#constraints). |
| dim0    | Input      | Input parameter. Index of the first dimension to be swapped, counted from 0. |
| dim1    | Input      | Input parameter. Index of the second dimension to be swapped, counted from 0. |

## Return Value

Returns a tensor with the same data type as the input, where the dimension positions of **dim0** and **dim1** are swapped.

## Constraints

1. **TileShape** has the same dimension count as the input and is used to tile **input**.

2. The value range of the input dimensions **dim0** and **dim1** is -D ≤ dim ≤ D-1, where D is the number of dimensions of **input**.

3. The current **Transpose** implementation has constraints and supports transposition only in the following scenarios:

   - 2 dimensions: any axes.
   - 3 dimensions: any axes.
   - 4 dimensions: Supported: axes 0 and 2, axes 1 and 3, axes 2 and 3, and axes 1 and 2. Not supported: axes 0 and 3, and axes 0 and 1.
   - 5D: Supported: axes 3 and 4. Other combinations are not supported.
   - Scenarios that do not require an actual transpose are directly supported: when **dim0** and **dim1** are the same, or when the input shape dimensions corresponding to **dim0** and **dim1** are both 1, the transpose result is equivalent to the input and is not subject to the 4D/5D axis combination constraints described above.

4. 32-byte alignment constraint on the last axis of the TileShape (only for scenarios that do not require an actual transpose): when transpose is determined to require no actual transpose and directly returns the input tensor (that is, when **dim0** == **dim1**, or when the shapes of both axes are 1), the byte count of the last dimension of the TileShape must be aligned to 32 bytes (**BLOCK_SIZE**). That is: `number of elements in the last dimension of the TileShape × sizeof(data type) % 32 == 0`.

    Note: In this scenario, the input tensor is directly returned (`return self`), but **TILE_REGISTER_COPY** in the subsequent flow still validates the 32-byte alignment of the last axis of the TileShape. If this constraint is not satisfied, the following error is reported: `CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0`.

5. For scenarios involving a transpose of the last axis, a temporary buffer must be reserved for data movement.

    Example:

    input: \[a, b, c, d\]  TileShape is \[t0, t1, t2, t3\]  data type is DT\_FP32

    dim0: 2

    dim1: 3

    The reserved temporary buffer is: t0 \* t1 \* align\(t2,16\) \* align\(t3,32 / sizeof\(DT\_FP32\)\)

6. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_HF8, DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, and DT_UINT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, and DT_UINT32
   <!-- end id6 -->
7. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same dimension count as the input.

Example 1: If the input shape is [m, n, p], **dim0** is 1, and **dim1** is 2, the output shape is [m, p, n]. If **TileShape** is set to [m1, n1, p1], then m1, n1, and p1 are used to tile the m, n, and p axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.transpose(x, 0, 1)
```

The results are as follows:

```python
Input data x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
Output data y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```
