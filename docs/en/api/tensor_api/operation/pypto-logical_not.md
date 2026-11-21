# pypto.logical\_not

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
<!-- npu="9030" id8 -->
- Kirin 9030 processors: Not supported
<!-- end id8 -->
<!-- npu="x90" id9 -->
- Kirin X90 processors: Not supported
<!-- end id9 -->
<!-- npu="9050" id10 -->
- Kirin 9050 processors: Not supported
<!-- end id10 -->

## Description

Converts zero values in the input tensor to True and non-zero values to False.

## Prototype

```python
logical_not(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns an output tensor of data type DT\_BOOL with the same shape as the source operand input.

## Constraints

1. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, and DT_INT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, and DT_UINT8
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, and DT_UINT8
   <!-- end id6 -->
2. TileShape must have the same dimensions as those of input.
3. Due to temporary memory usage, assume that TileShape is $[a,b,c,d]$, and let $N=a \cdot b \cdot c \cdot d$. When the input data type is DT_FP32, DT_FP16, or DT_BF16, the last dimension of the input tensor is 32-byte aligned and padded, and then the second-to-last dimension is row-padded before the tensor is copied to the UB. For an output tensor (data type: DT_BOOL), only the last dimension is 32-byte aligned and padded. When the input data type is DT_BOOL, DT_INT8, or DT_UINT8, the input tensor is not padded. Let $A=32/sizeof(input), d_{align}=CeilAlign(d, A), k=d_{align}/A, p=\lceil8/k\rceil, and c_{pad}=c+p-1$. When the input data type is DT_FP32, the TileShape size must satisfy $a \cdot b \cdot c_{pad} \cdot d_{align} \cdot \text{sizeof}(input) + a \cdot b \cdot c \cdot CeilAlign(d, 32) \cdot \text{sizeof}(BOOL) + 20.25\text{KB} < \text{UB}$. When the input data type is DT_FP16 or DT_BF16, the TileShape size must satisfy $a \cdot b \cdot c_{pad} \cdot d_{align} \cdot \text{sizeof}(input) + a \cdot b \cdot c \cdot CeilAlign(d, 32) \cdot \text{sizeof}(BOOL) + 12.54\text{KB} < \text{UB}$. When the input data type is DT_BOOL, DT_INT8, or DT_UINT8, the TileShape size must satisfy $N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 12.54\text{KB} < \text{UB}$.
   <!-- npu="950" id7 -->
   - Ascend 950PR&950DT products: If the input data type is DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, or DT_INT64, integer computation is used, where DT_INT16 or DT_UINT16 requires 4 KB of temporary memory, DT_INT32 or DT_UINT32 requires 8 KB of temporary memory, and DT_INT64 requires 16 KB of temporary memory. The TileShape size of the above types must satisfy $N \cdot \text{sizeof}(input) + N \cdot \text{sizeof}(BOOL) + 2048 \cdot \text{sizeof}(input) < \text{UB}$.
   <!-- end id7 -->
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
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

Result example:

```python
Input data x: [0 1 2 3 4]
Output data y: [True False False False False]
```
