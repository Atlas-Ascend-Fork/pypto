# pypto.transpose

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Supported
<!-- end id11 -->

## Description

Returns a tensor that is the transpose of the input tensor. The specified dimensions dim0 and dim1 are swapped.

## Prototype

```python
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 5 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).<br>The operator's support for shapes varies. For details, see the "Constraints" section.|
| dim0    | Input     | Input parameter. It is the index of the first dimension to be swapped, counted from 0.|
| dim1    | Input     | Input parameter. It is the index of the second dimension to be swapped, counted from 0.|

## Return Value

Returns a tensor of the same data type as the input, where the dimension positions of dim0 and dim1 are swapped.

## Constraints

1. TileShape has the same dimensions as the input and is used to tile the input.

2. The value range of input dimensions dim0 and dim1 is -D ≤ dim ≤ D-1, where D is the number of dimensions of input.

3. The current Transpose implementation has constraints and supports transposition only in the following scenarios:

   - 2D: any axis.
   - 3D: any axis.
   - 4D: axis 0 and axis 2, axis 1 and axis 3, axis 2 and axis 3, axis 1 and axis 2 supported; axis 0 and axis 3, axis 0 and axis 1 not supported.
   - 5D: axis 3 and axis 4 supported; other axes not supported.
   - Scenarios that do not require an actual transpose are directly supported: when dim0 and dim1 are the same, or when the input shape dimensions corresponding to dim0 and dim1 are both 1, the transpose result is equivalent to the input and is not subject to the 4D/5D axis combination constraints described above.

4. 32-byte alignment constraints along the last axis of the TileShape (only for scenarios that do not require an actual transpose): when transpose is determined to require no actual transpose and directly returns the input tensor (that is, when dim0==dim1, or when the shapes of both axes are 1), the byte count of the last dimension of TileShape must be aligned to 32 bytes (BLOCK_SIZE). This means `TileShape last dimension element count × sizeof(data type) % 32 == 0`.

    Note: In this scenario, the input Tensor (`return self`) is directly returned, but TILE_REGISTER_COPY in the subsequent flow still validates the 32-byte alignment of the last axis of TileShape. If this constraint is not satisfied, the error message `CHECK FAILED: lastDimBytes % BLOCK_SIZE == 0` is displayed.

5. In scenarios where the last axis needs to be transposed, a temporary space must be reserved for data movement.

    Example:

    input: \[a, b, c, d\]; TileShape: \[t0, t1, t2, t3\]; data type: DT\_FP32

    dim0: 2

    dim1: 3

    The reserved temporary space is t0 \* t1 \* align\(t2,16\) \* align\(t3,32 / sizeof\(DT\_FP32\)\).

6. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32, DT_HF8, DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, and DT_UINT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, and DT_UINT32
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: DT_FP16, DT_FP32, DT_INT32, and DT_INT16
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: DT_FP16, DT_FP32, DT_INT32, and DT_INT16
   <!-- end id10 -->
   <!-- npu="9050" id12 -->
   - Kirin 9050 processors: DT_FP16, DT_FP32, DT_INT32, and DT_INT16
   <!-- end id12 -->
7. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input.

Example 1: If the input shape is [m, n, p], dim0 is 1, dim1 is 2, the output is [m, p, n], and TileShape is set to [m1, n1, p1], then m1, n1, and p1 are used to tile the m, n, and p axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.transpose(x, 0, 1)
```

Result example:

```python
Input data x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
Output data y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```
