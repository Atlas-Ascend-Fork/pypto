# pypto.cos

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

Computes the cosine value (trigonometric function cos\(\)) of each element in the input tensor. It supports element-wise operations and returns a tensor with the same shape as the input.

## Prototype

```python
cos(input: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported data types: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor with the same shape and data type as the input tensor. Each element of the output tensor is the cosine value of the corresponding element in the input tensor.

## Constraints

1. The input and output tensors must have the same type.
2. Due to temporary memory usage, the TileShape size has additional constraints. If TileShape is set to \[a,b,c,d\], then 5\*a\*b\*c\*d\*sizeof\(DT_FP32\) < UB.
3. The supported input range is [-65504.0,65504.0].
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.cos(x)
```

Result example:

```python
Input data x: [0.0000, 0.7854, 1.5708, 2.3562]
Output data y: [1.0000, 0.7071, 0.0000, -0.7071]
```
