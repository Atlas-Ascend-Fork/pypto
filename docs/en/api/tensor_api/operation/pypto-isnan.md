# pypto.isnan

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

Determines whether each element in a tensor is NaN (not a number) element-wise.

If an element is NaN or -NaN, its value at the corresponding position in the result is True; other elements are False.

Common finite values, `+0.0`/`-0.0`, and `+inf`/`-inf` are all regarded as non-NaN, and their values at the corresponding positions are False.

## Prototype

```python
isnan(self: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| self   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports 1 to 4 dimensions. The number of elements in the shape must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is Boolean type DT_BOOL, and its shape size is the same as that of the input tensor.

## Constraints

1. Only data types such as DT_FP16, DT_BF16, and DT_FP32 are supported.
2. The last axis of TileShape and ViewShape must be 32-byte aligned with the output tensor type. Since the output tensor is of the Boolean type, the last axis of TileShape and ViewShape must be a multiple of 32.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
4. Considering the input, output, and temporary storage space usage, the TileShape size must meet additional constraints. When an input tensor is copied to the UB, the last dimension is 32-byte aligned and padded, and then the second-to-last dimension is row-padded. For an output tensor (data type: DT\_BOOL), only the last dimension is 32-byte aligned and padded. Assume that TileShape is \[a,b,c,d\] (where d is a multiple of 32 according to item 2 in the Constraints section), and let $A=32/sizeof(self), k=d/A, p=\lceil8/k\rceil, and c_{pad}=c+p-1$; then the total UB space usage is:

   $$
   a*b*c_{pad}*d*sizeof(self)+a*b*c*d*sizeof(DT\_BOOL)+3*c*d*sizeof(DT\_FP32)+32 <= UB
   $$

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
self = pypto.tensor([3, 3], pypto.DT_FP32)
out = pypto.isnan(self)
```

Result example:

```python
Input data self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
Output data out: [[False True False],
             [False False False],
             [False False False]]
```
