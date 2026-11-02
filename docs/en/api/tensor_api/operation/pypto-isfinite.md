# pypto.isfinite

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:54:04.108Z pushedAt=2026-10-28T11:57:29.962Z nl=2 -->

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

Determines whether each element in a tensor is a finite value.

For integer-type tensors, returns a boolean tensor of the same shape as the input, with all elements set to `True`.

For floating-point tensors, only `inf`, `-inf`, and `nan` are considered non-finite; the corresponding positions in the result are `False`, while all other positions are `True`.

## Prototype

```python
isfinite(self: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, and DT_INT64.<br>Empty tensors are not supported. The shape size supports 1 to 4 dimensions. The shape size must not exceed **2147483647** (**INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type is the Boolean type DT_BOOL and whose shape size is the same as that of the input tensor.

## Constraints

1. Only the following data types are supported: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, and DT_INT64.
2. The last axis of both TileShape and ViewShape must be 32 byte-aligned with respect to the output tensor's data type. Since the output tensor is of boolean type, the last axis of TileShape and ViewShape must be a multiple of 32.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n] and the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 32)
```

### API Call Example

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isfinite(self)
```

The results are as follows:

```python
Input data self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
Output data out: [[True False True],
             [False True True],
             [True True False]]
```
