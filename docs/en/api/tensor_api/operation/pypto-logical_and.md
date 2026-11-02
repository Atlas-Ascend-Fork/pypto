# pypto.logical\_and

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:00:52.735Z pushedAt=2026-10-28T11:57:29.986Z nl=2 -->

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

Performs an element-wise logical AND operation on two input tensors. Operation rules:

- If the input tensors are of the bool type, **True** and **True** -\> **True**; otherwise, the result is **False**.
- If the input tensors are numeric, they are automatically converted to **True**/**False**, where 0 is **False** and any non-zero value is **True**.

## Prototype

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Input tensors of different data types are supported. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |
| other   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, and DT_INT32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. Input tensors of different data types are supported. Multi-dimensional broadcasting to the same shape is supported. The shape size must not exceed 2147483647 (that is, INT32_MAX). |

## Return Value

Returns the output tensor, whose data type is DT\_BOOL and whose shape is the broadcast shape.

## Constraints

1. The TileShape must have the same dimension count as `input` and `other`.
2. Due to temporary memory usage, the TileShape size has additional constraints. Assuming that the TileShape is [a,b,c,d], then a*b*c*d*sizeof(input) + a*b*c*d*sizeof(other) + a*b*c*d*sizeof(BOOL) + 1.1875KB must be less than the available memory upper limit.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1: In a non-broadcast scenario, if the input shape is [m, n], **other** is [m, n], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: In a broadcast scenario, if the input shape is [m, n], **other** is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

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

The results are as follows:

```python
Input data x:  [True, False]
Input data y1: [True, True]
Input data y2: [[True, False], [False, True]]
Output data z1: [True, False]
Output data z2: [[True, False], [False, False]]
```
