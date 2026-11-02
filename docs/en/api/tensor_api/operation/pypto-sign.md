# pypto.sign

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:32:27.228Z pushedAt=2026-10-28T11:57:30.098Z nl=2 -->

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

Computes the sign value of each element in the input tensor, element by element. The computation formula is as follows:

$$
\text{sign}(x) = \begin{cases}
-1 & \text{if } x < 0 \\
0 & \text{if } x = 0 \\
1 & \text{if } x > 0
\end{cases}
$$

## Prototype

```python
sign(input: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: **Tensor**.<br>The data types supported by Tensor vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, **INT32_MAX**). |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the sign values (-1, 0, or 1) of the corresponding elements of the input tensor.

## Constraints

1. The TileShape must have the same dimension count as the input.
2. Because temporary memory is used, when setting TileShape, ensure that the total usage of the input tile, output tile, and temporary space is less than the available UB. Assume that the last two dimensions of TileShape are `H` and `W` (for a one-dimensional TileShape, `H = 1`). The temporary space for each data type is as follows:

   | Input Data Type | Actual Working Data Type | Tail Axis Alignment Length | Temporary Space Size (Bytes) |
   |--------------|------------------|--------------|----------------------|
   | `DT_INT8` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `H * W_align * sizeof(DT_FP16)` |
   | `DT_FP16` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `2 * H * W_align * sizeof(DT_FP16) + 32` |
   | `DT_BF16` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_FP32` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_INT16`, `DT_INT32`, `DT_INT64` | Same as input | Not applicable | `32` |

   The two equal-sized temporary spaces in the floating-point computation path are used for working data and the comparison mask, respectively, and the additional 32 bytes are used for a scalar temporary block. A `DT_BF16` input is converted to `DT_FP32` through AutoCast before entering the Sign TileOp, so temporary space is allocated according to the `DT_FP32` path. After the computation is complete, the result is converted back to `DT_BF16`.
3. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, DT_INT64.
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32.
   <!-- end id6 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: If the input shape is [m, n] and the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sign(x)
```

The results are as follows:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [-1.0, 0.0, 1.0, 1.0, -1.0]
```
