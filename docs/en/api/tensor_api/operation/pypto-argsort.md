# pypto.argsort

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:00:09.153Z pushedAt=2026-10-28T11:57:29.751Z nl=2 -->

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

Obtains the indices of the input after sorting along the specified axis in ascending or descending order.

## Prototype

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=True) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|------------------------------------------------------------------------------|
| input     | Input        | Source operand.<br>Supported type: **Tensor**.<br>Supported data types of Tensor: DT_FP32 and DT_FP16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| dim       | Input        | Dimension along which to sort.<br>Negative indices are supported, with a value range of [-input.dim, input.dim - 1]. The default value is None, which indicates the last axis. For four-dimensional input, axis 0 (dim=0 or dim=-4) is not supported. |
| descending | Input       | If **True**, returns the indices in descending order. If **False**, returns the indices in ascending order. |

## Return Value

Returns a **Tensor** containing the indices of the input after sorting along the **dim** axis according to **descending**.

## Constraints

1. Currently, splitting **ViewShape** along the **dim** axis is not supported, which means **ViewShape[dim]** must equal **InputShape[dim]**.
2. Currently, **TileShape** along the **dim** axis must be a multiple of 32, which means **TileShape[dim]** % 32 = 0.
3. When the shape is large $(\frac{TileShape\ Size}{TileShape[dim]} * CeilAlign(ViewShape[dim], 32) >= 6144)$, the number of tile splits along the sorting axis must be less than 128.
4. For four-dimensional input, sorting along axis 0 is not supported.
5. When equal values are encountered during sorting, stable sorting is used to return the corresponding indices.
6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same dimension count as the input.

For example, if the input shape is [m, n, p], **dim** is 2, and **descending** is **True**, the output shape is [m, n, p]. If **TileShape** is set to [m1, n1, p1], then m1, n1, and p1 are used to split the m, n, and p axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### API Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

The results are as follows:

```python
Input data x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
Output data y: [[2, 1, 0],
            [2, 1, 0]]
```
