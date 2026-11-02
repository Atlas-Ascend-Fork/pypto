# pypto.logical\_not

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:01:36.669Z pushedAt=2026-10-28T11:57:29.989Z nl=2 -->

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

Converts zero values in the input tensor to `True` and all non-zero values to `False`.

## Prototype

```python
logical_not(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                          |
|----------------|--------------|----------------------------------------------------------------------|
| input          | Input        | Source operand.<br>Supported type: Tensor.<br>The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (that is, INT32_MAX). |

## Return Value

Returns an output tensor of data type DT\_BOOL with the same shape as the source operand `input`.

## Constraints

1. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64.
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8.
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8.
   <!-- end id6 -->
2. The TileShape must have the same dimension count as the input.
3. Because temporary memory is used, assume that the TileShape is $[a,b,c,d]$ and let $N=a \cdot b \cdot c \cdot d$. When the input data type is DT_FP32, the TileShape size must satisfy $N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 20.25\text{KB} < \text{UB}$; for other basic supported types, it must satisfy $N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 12.54\text{KB} < \text{UB}$.
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: When the input data type is DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, or DT_INT64, integer computation is used. Among them, DT_INT16/DT_UINT16 require 4KB of temporary memory, DT_INT32/DT_UINT32 require 8KB of temporary memory, and DT_INT64 requires 16KB of temporary memory. The TileShape size of the above types must satisfy $N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 2048 \cdot \text{sizeof}(input) < \text{UB}$.
   <!-- end id7 -->
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
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

The results are as follows:

```python
Input data x: [0 1 2 3 4]
Output data y: [True False False False False]
```
