# pypto.ne

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
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Supported
<!-- end id11 -->

## Description

Performs element-wise not-equal comparison operations.

## Prototype

```python
ne(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor. The supported tensor data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Source operand.<br>Supported types: Tensor, float, and Element.<br>When the type is float, it is automatically converted to the Element type, where float corresponds to DT_FP32. To use other data types, construct the value through Element.<br>Supported Tensor and Element data types vary by model. For details, see [Constraints](#constraints).<br>The data types of the two source operands must be the same.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns a tensor. Its shape is the same as that of the input tensor and its data type is DT\_BOOL. If the element value at a position in input does not equal the element value at the corresponding position in other, the return value at that position is True; otherwise, it is False.

## Constraints

1. The types of input and other must be the same.
2. Multi-dimensional broadcasting to the same shape is supported.
3. Tensor and Element data types are supported as follows:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_FP16, DT_FP32, DT_INT16, DT_INT64, and DT_UINT64
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_FP16 and DT_FP32
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_FP16 and DT_FP32
   <!-- end id6 -->
   <!-- npu="9030" id9 -->
   - Kirin 9030 processors: DT_FP16 and DT_FP32
   <!-- end id9 -->
   <!-- npu="x90" id10 -->
   - Kirin X90 processors: DT_FP16 and DT_FP32
   <!-- end id10 -->
   <!-- npu="9050" id12 -->
   - Kirin 9050 processors: DT_FP16 and DT_FP32
   <!-- end id12 -->
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
5. UB space usage of the input tensor: When other is a tensor, the last dimension of both the input and other tensors is 32-byte aligned and padded, and then the second-to-last dimension is row-padded when they are copied to the UB. When other is a scalar (float or Element), only the input tensor is padded as described above, and the scalar other is not padded into a tensor. Assume that TileShape is \[a,b,c,d\], and let $A=32/sizeof(input), d_{align}=CeilAlign(d, A), k=d_{align}/A, p=\lceil8/k\rceil, and c_{pad}=c+p-1$. Then the actual UB space usage of each input tensor is $a*b*c_{pad}*d_{align}*sizeof(input)$. For the output tensor (data type: DT\_BOOL), only the last dimension is 32-byte aligned. The UB space occupied by the output tensor is $a*b*c*CeilAlign(d, 32)*sizeof(DT\_BOOL)$. In addition, after TileShape is set, a fixed temporary space irrelevant to the TileShape size is required for tiled execution. The size of the temporary space cannot exceed 12,608 bytes. To sum up, when other is a tensor, the total UB space usage must meet the following requirements:

   $$
   2*a*b*c_{pad}*d_{align}*sizeof(input)+a*b*c*CeilAlign(d, 32)*sizeof(DT\_BOOL)+12608 <= UB
   $$

   When other is a scalar, the total UB space usage must meet the following requirements:

   $$
   a*b*c_{pad}*d_{align}*sizeof(input)+a*b*c*CeilAlign(d, 32)*sizeof(DT\_BOOL)+12608 <= UB
   $$

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], other is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], other is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.ne(a, b)
```

Result example:

```python
Input data a: [1.0 2.0 3.0]
Input data b: [2.0 2.0 2.0]
Output data out: [True, False, True]
```
