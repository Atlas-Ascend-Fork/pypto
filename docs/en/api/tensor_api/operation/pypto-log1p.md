# pypto.log1p

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

Performs a base-e logarithmic operation on 1 + input.

## Prototype

```python
log1p(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input    | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32, DT_FP16, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## TileShape Setting Example

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Constraints

1. Considering the input, output, and temporary storage space usage, the TileShape size must meet additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 8)$. The total UB space usage is as follows:
   $$
   6*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log1p(x)
```

Result example:

```python
Input data x: [1e-99]
Output data y: [1e-99]
```
