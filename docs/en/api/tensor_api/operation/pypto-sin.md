# pypto.sin

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:34:03.090Z pushedAt=2026-10-28T11:57:30.102Z nl=2 -->

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

Computes the sine (trigonometric function sin\( \)) of each element in the input tensor, operating element-wise.

## Prototype

```python
sin(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data types: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the sine values of the corresponding elements of the input tensor.

## Constraints

1. The input tensor and the output tensor must have the same type.
2. Due to temporary memory usage, the TileShape size has an additional constraint. Assuming the TileShape is \[a,b,c,d\], then 5\*a\*b\*c\*d\*sizeof\(DT_FP32\) < UB.
3. The supported input range is [-65504.0, 65504.0].
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sin(x)
```

The results are as follows:

```python
Input data x: [-0.5461, 0.1347, -2.7266, -0.2746]
Output data y: [-0.5194, 0.1343, -0.4032, -0.2711]
```
