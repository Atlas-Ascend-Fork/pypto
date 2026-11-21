# pypto.isfinite

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

Determines whether each element in a tensor is a finite value.

If the tensor is of integer type, this API returns a boolean tensor of the same shape as the input, with all elements set to True.

If the tensor is of floating-point type, only inf, nan, and -inf are considered non-finite, with each element at the corresponding position in the result being False; all other elements are True.

## Prototype

```python
isfinite(self: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| self   | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, and DT_INT64.<br>Empty tensors are not supported. The shape supports 1 to 4 dimensions. The number of elements in the shape must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is Boolean type DT_BOOL, and its shape size is the same as that of the input tensor.

## Constraints

1. Only the following data types are supported: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, and DT_INT64.
2. The last axis of TileShape and ViewShape must be 32-byte aligned with the output tensor type. Since the output tensor is of the Boolean type, the last axis of TileShape and ViewShape must be a multiple of 32.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 32)
```

### API Call Example

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isfinite(self)
```

Result example:

```python
Input data self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
Output data out: [[True False True],
             [False True True],
             [True True False]]
```
