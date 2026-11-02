# pypto.prod

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:17:48.727Z pushedAt=2026-10-28T11:57:30.043Z nl=2 -->

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

Performs cumulative multiplication on a multidimensional vector along the specified dimension.

## Prototype

```python
prod(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_INT32, and DT_INT16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim     | Input      | Dimension for reduction.<br>Any single axis is supported. |
| keepdim | Input      | Whether to retain the reduced dimension after reduction.<br>The default value is **False**. |

## Return Value

Returns the output tensor. The shape of the output tensor depends on the **keepdim** parameter.

If the **keepdim** parameter is **True**, the reduced dimension is retained after the reduction operation. The shape of the output tensor is the same as that of the input tensor in all dimensions except the dimension specified by **dim**, where the size is **1**.

If the **keepdim** parameter is **False** (default), the reduced dimension is removed from the output tensor, while the corresponding dimension in the TileShape remains unchanged. Therefore, it is recommended to reset the TileShape before calling other operations.

## Constraints

1. The TileShape size must not exceed 64 KB.

2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## TileShape Setting Example

The TileShape must have the same dimension count as the input.

For example, if the input shape is [m, n], the output is [m, 1], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.prod(x, -1, True)
```

The results are as follows:

```txt
Input data x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data y: [[6.0],
             [6.0]]
```
