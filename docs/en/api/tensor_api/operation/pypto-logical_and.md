# pypto.logical\_and

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

Performs an element-wise logical AND operation on two input tensors. Operation rules are as follows:

- If the input tensors are of the bool type, True and True -> True; otherwise, the result is False.
- If the input tensor values are numeric, they are automatically converted to True/False, where 0 is False and any non-zero value is True.

## Prototype

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The input tensors can be of different data types. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| other   | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The input tensors can be of different data types. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type is DT\_BOOL, and its shape is the broadcast shape.

## Constraints

1. TileShape must have the same dimensions as those of input and other.
2. Considering the input, output, and temporary storage space usage, the TileShape size must meet additional constraints. Assume that TileShape is [a,b,c,d], and let $d_{align}=CeilAlign(d, 32)$, $E=c*d_{align}+CeilAlign(d_{align},128)-d_{align}$, and $M=CeilAlign(\lceil d_{align}/8 \rceil, 32)$. $compute\_dtype$ is the compute precision (DT_FP32 if either input or other is DT_FP32, DT_BF16, or DT_INT32, otherwise DT_FP16). The total UB space usage is computed as follows:
   $$
   a*b*c*d*(sizeof(input)+sizeof(other)+sizeof(DT\_BOOL)) + 2*E*sizeof(compute\_dtype) + max(c*M, E*sizeof(DT\_FP16)) + 32 <= UB
   $$
   If $compute\_dtype$ is DT_FP16, $E * sizeof(DT\_FP16)$ is 0.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
x = pypto.tensor([2], pypto.DT_BOOL)
y1 = pypto.tensor([2], pypto.DT_BOOL)
z1 = pypto.logical_and(x, y1)
# Broadcasting is supported.
y2 = pypto.tensor([2,2], pypto.DT_BOOL)
z2 = pypto.logical_and(x, y2)
```

Result example:

```python
Input data x:  [True, False]
Input data y1: [True, True]
Input data y2: [[True, False], [False, True]]
Output data z1: [True, False]
Output data z2: [[True, False], [False, False]]
```
