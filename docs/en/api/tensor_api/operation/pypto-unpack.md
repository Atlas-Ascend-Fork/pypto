# pypto.unpack

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:45:25.643Z pushedAt=2026-10-28T11:57:30.148Z nl=2 -->

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

Unpacks data by reinterpreting the input tensor of the UINT8 type as an output tensor of the specified data type.

## Prototype

```python
unpack(self: Tensor, dstDataType: DataType) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------------|-----------|----------------------------------------------------------------------|
| self | Input | Source operand.<br>Supported data type: DT_UINT8.<br>Empty tensors are not supported. The shape supports only one dimension. Supported data format: ND. |
| dstDataType | Input | Specifies the output data type.<br>Supported data types: DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_BF16, DT_FP16, DT_FP32. |

## Return Value

Returns a tensor. Its shape is one-dimensional, its data type is the type specified by `dstDataType`, and its element count is the number of elements in the input tensor divided by the byte size of `dstDataType`. That is, the contiguous bytes of the input tensor are reinterpreted as elements of the corresponding type according to the byte width of `dstDataType`.

## Constraints

1. The sizes of InputShape, ValidShape, and TileShape must be divisible by the byte size of `dstDataType`.
2. The shape size must not exceed 2147483647 (that is, INT32_MAX).
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through `set_vec_tile_shapes`.

The TileShape dimensions must be consistent with the input.

If the input shape is [n], the output shape is [m] (where m = n / sizeof(dstDataType)), and TileShape is set to [t], then t is used to split the input axis n.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

```python
x = pypto.tensor([6, 5, 4, 3, 8, 6, 5, 1], pypto.DT_UINT8)
y = pypto.unpack(x, pypto.DT_INT32)
```

The results are as follows:

```python
Input data x: [  6  5  4  3  8  6  5  1  ]    # DT_UINT8, shape=[8]
Output data y: [ 0x03040506 0x01050608 ]        # DT_INT32, shape=[2]
```

Note: Every 4 bytes of the 8 input UINT8 elements are interpreted as 1 INT32 element, so the 8 UINT8 elements are unpacked into 2 INT32 elements.
