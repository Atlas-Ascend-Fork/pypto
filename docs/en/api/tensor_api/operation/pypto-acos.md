# pypto.acos

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

Computes the arccosine of each element in the input tensor, performing element-wise operations. The result falls within the interval $[0, \pi]$. When the absolute value of an input element is greater than 1, the corresponding output is NaN.

$$
y_i = \arccos(x_i)
$$

## Prototype

```python
acos(input: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description|
|--------|-----------|------|
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the arccosine values of the corresponding elements of the input tensor.

## Constraints

1. Considering the input, output, and temporary storage space usage, the TileShape size must meet additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 64)$. The total UB space usage is as follows:
   $$
   6*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.acos(x)
```

Result example:

```python
Input data x: [-1.0000, 0.0000, 1.0000]
Output data y: [3.1416, 1.5708, 0.0000]
```
