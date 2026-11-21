# pypto.cbrt

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

Computes the cube root of each element in the input tensor element-wise and returns a tensor with the same shape as the input. The computation formula is as follows:

$$
y_i = \sqrt[3]{x_i}
$$

## Prototype

```python
cbrt(input: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. The input and output tensors must have the same type.
2. Considering the input, output, and temporary storage space usage, the TileShape size must meet additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 32/sizeof(dt))$. The total UB space usage is as follows:

   $$
   4*a*b*c*d_{align}*sizeof(dt) <= UB
   $$

   where $dt$ is the data type of the input tensor.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([1.0, 2.0], pypto.DT_FP32)
y = pypto.cbrt(x)
```

Result example:

```python
Input  x:[[8.0, -8.0]]
Output y:[[2.0, -2.0]]
```
