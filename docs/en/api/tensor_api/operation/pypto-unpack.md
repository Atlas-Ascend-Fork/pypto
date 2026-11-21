# pypto.unpack

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Unpacks data by reinterpreting the input tensor of the UINT8 type as an output tensor of the specified data type.

## Prototype

```python
unpack(self: Tensor, dstDataType: DataType) -> Tensor
```

## Parameters

| Parameter      | Input/Output| Description                                                                |
|--------------|-----------|----------------------------------------------------------------------|
| self         | Input     | Source operand.<br>Supported data type: DT_UINT8.<br>Empty tensors are not supported. The shape supports only 1 dimension. The supported data format is ND.|
| dstDataType  | Input     | Output data type.<br>The supported data types are DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64, DT_BF16, DT_FP16, and DT_FP32.|

## Return Value

Returns a tensor. Its shape is 1D, its data type is the type specified by `dstDataType`, and its element count is the number of elements in the input tensor divided by the byte size of `dstDataType`. That is, the contiguous bytes of the input tensor are reinterpreted as elements of the corresponding type based on the byte width of `dstDataType`.

## Constraints

1. The sizes of InputShape, ValidShape, and TileShape must be exactly divisible by the byte size of `dstDataType`.
2. The shape size must not exceed 2147483647 (INT32_MAX).
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through `set_vec_tile_shapes`.

TileShape must have the same dimensions as the input.

For example, if the input shape is [n], the output is [m] (where m = n / sizeof(dstDataType)), and TileShape is set to [t], then t is used to tile the input axis n.

```python
pypto.set_vec_tile_shapes(16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

```python
x = pypto.tensor([8], pypto.DT_UINT8)
y = pypto.unpack(x, pypto.DT_INT32)
```

When the caller provides the following input data, the result is as follows:

```python
Input data x: [  6  5  4  3  8  6  5  1  ]    # DT_UINT8, shape=[8]
Output data y: [ 0x03040506 0x01050608 ]        # DT_INT32, shape=[2]
```

Note: Every four bytes of the eight input UINT8 elements are interpreted as one INT32 element, so the eight UINT8 elements are unpacked into two INT32 elements.
