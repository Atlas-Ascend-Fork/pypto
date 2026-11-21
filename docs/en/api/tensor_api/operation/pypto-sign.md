# pypto.sign

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->

## Description

Computes the sign value of each element in the input tensor, performing element-wise operations. The computation formula is as follows:

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

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and its elements are the sign values (-1, 0, or 1) of the corresponding elements of the input tensor.

## Constraints

1. TileShape must have the same dimensions as those of input.
2. Due to the use of temporary memory, when setting TileShape, ensure that the total size of the input tile, output tile, and temporary space is less than the available UB. Assume that the last two dimensions of TileShape are `H` and `W` (for one-dimensional TileShape, use `H = 1`). The temporary space for each data type is as follows:

   | Input Data Type| Actual Working Data Type| Last Axis Alignment Length| Temporary Space Size (Bytes)|
   |--------------|------------------|--------------|----------------------|
   | `DT_INT8` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `H * W_align * sizeof(DT_FP16)` |
   | `DT_FP16` | `DT_FP16` | `W_align = CeilAlign(W, 16)` | `2 * H * W_align * sizeof(DT_FP16) + 32` |
   | `DT_BF16` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_FP32` | `DT_FP32` | `W_align = CeilAlign(W, 8)` | `2 * H * W_align * sizeof(DT_FP32) + 32` |
   | `DT_INT16`, `DT_INT32`, `DT_INT64`| Same as the input| N/A| `32` |

   The floating-point computation path uses two equal-sized temporary spaces: one for working data and one for the comparison mask. The extra 32 bytes store the scalar temporary block. The `DT_BF16` input is converted to `DT_FP32` through AutoCast before entering the Sign TileOp, so temporary space is allocated based on the `DT_FP32` path. After the computation is complete, the result is converted back to `DT_BF16`.
3. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32, and DT_INT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, and DT_INT32
   <!-- end id6 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sign(x)
```

Result example:

```python
Input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
Output data y: [-1.0, 0.0, 1.0, 1.0, -1.0]
```
