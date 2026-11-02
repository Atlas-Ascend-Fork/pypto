# pypto.log1p

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:59:25.128Z pushedAt=2026-10-28T11:57:29.979Z nl=2 -->

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

Computes the natural logarithm (base e) of 1 + **input**.

## Prototype

```python
log1p(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input    | Input      | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of a tensor: DT_FP32, DT_FP16, DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns the output tensor, whose data type is the same as that of **input** and whose shape is the same as that of **input**.

## TileShape Setting Example

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Constraints

1. Considering the input, output, and temporary storage space usage, the TileShape size has additional constraints. Assume that TileShape is \[a,b,c,d\], and let $d_{align}=CeilAlign(d, 8)$. Then the total UB space usage is:
   $$
   6*a*b*c*d_{align}*sizeof(DT\_FP32) <= UB
   $$
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log1p(x)
```

The results are as follows:

```python
Input data x: [1e-99]
Output data y: [1e-99]
```
