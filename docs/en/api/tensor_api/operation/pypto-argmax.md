# pypto.argmax

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:58:43.766Z pushedAt=2026-10-28T11:57:29.745Z nl=2 -->

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

Computes the index of the maximum value of a multidimensional vector along a specified dimension.

Defines the specified computation dimension as the R-axis (Reduce axis) and the non-specified dimension as the A-axis (Normal axis). For a two-dimensional matrix of shape \(2, 3\), when computing the indices of the maximum values along the first dimension, the output is \[1, 1, 1\]; when computing along the second dimension, the output is \[2, 2\].

## Prototype

```python
argmax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim     | Input      | Dimension for reduction.<br>Any single axis is supported.                                       |
| keepdim | Input      | Whether to retain the reduced dimension after reduction.<br>The default value is **False**. |

## Return Value

Returns the output tensor. The data type of the output tensor is DT_INT32, and its shape depends on the **keepdim** parameter.

If the **keepdim** parameter is **True**, the reduced dimension is retained after the reduction operation. The shape of the output tensor is the same as that of the input tensor in all dimensions except the dimension specified by **dim**, where the size is **1**.

If the **keepdim** parameter is **False** (default), the reduced dimension is removed from the output tensor, while the corresponding dimension in the TileShape remains unchanged. Therefore, it is recommended to reset the TileShape before calling other operations.

## Constraints

1. The TileShape size must not exceed 64 KB.

2. The last axis must be 32-byte aligned.

3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same dimension count as the input.

For example, if the input shape is [m, n], the output is [m, 1], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Note: If **keepdim** is set to **false**, the reduced dimension is removed from the output tensor, while the corresponding dimension in the TileShape remains unchanged. Therefore, it is recommended to reset the TileShape before calling other operations.

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argmax(x, -1, True)
```

The results are as follows:

```python
Input data x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output data y: [[2],
             [2]]
```
