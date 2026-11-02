# pypto.expand\_clone

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:30:49.596Z pushedAt=2026-10-28T11:57:29.870Z nl=2 -->

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

Broadcasts the input tensor along axes of size 1 to match the specified shape, and returns a new tensor with actual memory allocation.

## Prototype

```python
expand_clone(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-------------|-----------|----------------------------------------------------------------------|
| input       | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_BF16, DT_FP32, DT_FP16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, and DT_BOOL.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The size of the broadcast axis must be 1. The shape size must not exceed **2147483647** (**INT32_MAX**). |
| shape       | Input      | Target shape.<br>Supported data type: List[int].<br>The shape size must not exceed INT32_MAX. The number of shape dimensions must be consistent with that of the input, and except for the broadcast axis, the size of other axes must be equal to the corresponding shape of input. |
| valid_shape | Input      | Keyword parameter.<br>Source operand, used to define the dynamic shape of the output tensor. It is a keyword parameter used for dynamic graphs and can be omitted for static graphs.<br>Supported types: List[SymbolicScalar] and List[int]. |

## Return Value

Returns an output tensor with the same data type as `input` and the specified shape.

## Constraints

1. Multi-dimensional broadcasting is supported. For axes along which the input tensor is to be broadcast, the corresponding shape size must be 1.
2. The ViewShape of the input has the same dimensionality as the input, with `viewshape[dim] = 1` and `input[dim] = 1`, where `dim` denotes the axis to be expanded. No restrictions apply to other dimensions. Examples are as follows:
    1. \[a,1\] is expanded to \[a,5\], where dim=1, indicating that expansion is performed on dim 1.
    2. len\(viewshape\)=2 and viewshape\[dim\]=1

3. Notes on **valid\_shape**:

    In the dynamic graph scenario, assume that the tensor input \[a,1\] is expanded to \[a,5\], and the ViewShape is set to \[a,2\]. The framework generates \[a,2\] tiles through the pypto.loop loop and concatenates them by offset. In this case, if **valid\_shape** is not passed, the code generates a tensor that is entirely \[a,2\] by default (for example, pypto.expand\_clone\(input, \[a,2\]\)).

    However, when the total size \[a,5\] is not divisible by the tile size \[a,2\], the valid shape of the tail tile (for example, \[a,1\]) cannot be automatically inferred by the framework. For example, the last column may contain only one element instead of a complete \[a,2\] tile. In this case, the actual valid shape of the tail tile must be explicitly specified through **valid\_shape**, as follows:

    pypto.expand\_clone\(input, \[a,2\], valid\_shape = \[a, pypto.min\(2, 5 - 2 \* b\_idx\),\)

    Here, **b\_idx** represents the loop index.

4. The dimensions of **tileshape** are the same as those of **result**, and **tileshape** is used to split **result**.
5. There are no additional constraints on the size and shape of **tileshape**, except that it must not exceed the **ub size**.

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

For example, if the input shape is [m, 1], the output is [m, n], and the TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

```python
# static graph
a = pypto.tensor([1,8], pypto.DT_INT32)
out1 = pypto.expand_clone(a, [4,8])
# dynamic graph
out2 = pypto.expand_clone(a, [4,8], valid_shape = [pypto.symbolic_scalar(4), pypto.symbolic_scalar(8)])
```

The results are as follows:

```python
Input data a:     [[1, 2, 3, 4, 5, 6, 7, 8]]
Output data out1:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
Output data out2:  [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]]
```
