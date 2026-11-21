# pypto.argsort

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

Obtains the indices of the input sorted in ascending or descending order along the specified axis.

## Prototype

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=False) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension along which to sort.<br>Negative indices are supported. The value range is [-input.dim, input.dim - 1]. The default value is None, indicating the last axis. A four-dimensional input does not support axis 0 (dim=0 or dim=-4).|
| descending | Input     | True: returns the indices in descending order. False: returns the indices in ascending order. The default value is False, indicating that the indices are returned in ascending order.|

## Return Value

Returns a tensor, containing the indices of the input sorted in descending order along the dim axis.

## Constraints

1. Currently, tiling ViewShape along the dim axis is not supported, which means ViewShape[dim] must equal InputShape[dim].
2. Currently, TileShape along the dim axis must be a multiple of 32, which means TileShape[dim] % 32 = 0.
3. When the shape is large $(\frac{TileShape\ Size}{TileShape[dim]} * CeilAlign(ViewShape[dim], 32) >= 6144)$, the number of tiles along the sorting axis must be less than 128.
4. For a four-dimensional input, sorting along axis 0 is not supported.
5. When there are equal values during sorting, stable sorting is used to return the corresponding indices.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the input.

For example, if the input shape is [m, n, p], dim is 2, descending is True, the output is [m, n, p], and TileShape is set to [m1, n1, p1], then m1, n1, and p1 are used to tile the m, n, and p axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

Result example:

```python
Input data x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
Output data y: [[2, 1, 0],
            [2, 1, 0]]
```
