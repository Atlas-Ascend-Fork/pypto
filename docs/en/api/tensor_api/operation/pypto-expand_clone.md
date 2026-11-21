# pypto.expand\_clone

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

| Parameter     | Input/Output| Description                                                                |
|-------------|-----------|----------------------------------------------------------------------|
| input       | Input     | Source operand.<br>Supported type: Tensor.<br>A tensor supports the following data types: DT_BF16, DT_FP32, DT_FP16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, and DT_BOOL.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size of the broadcast axis must be 1. The shape size must not exceed 2147483647 (INT32_MAX).|
| shape       | Input     | Target shape.<br>The supported data type is List[int].<br>The shape size must not exceed INT32_MAX. The shape dimensions must be the same as those of the input. Except for the broadcast axis, the size of other axes must be the same as that of the input shape.|
| valid_shape | Input     | Keyword parameter.<br>Source operand, which is used to define the dynamic shape of the output tensor. It is a keyword parameter used for dynamic graphs and can be omitted for static graphs.<br>The supported types are List[SymbolicScalar] and List[int].|

## Return Value

Returns an output tensor with the same data type as input and the specified shape.

## Constraints

1. Multi-dimensional broadcasting is supported. For axes along which the input tensor is to be broadcast, the corresponding shape size must be 1.
2. viewshape of the input has the same dimension as the input, with viewshape\[dim\]=1 and input\[dim\]=1, where dim denotes the axis to be expanded. No restrictions apply to other dimensions. Examples are as follows:
    1. \[a,1\] is expanded to \[a,5\], where dim=1, indicating that expansion is performed on dim 1.
    2. len\(viewshape\)=2 and viewshape\[dim\]=1 must be satisfied.

3. Notes on valid\_shape:

    In the dynamic graph scenario, assume that the tensor input \[a,1\] is expanded to \[a,5\] and ViewShape is set to \[a,2\]. The framework generates the \[a,2\] tiles through the pypto.loop loop and concatenates them by offset. In this case, if valid\_shape is not passed, the code will generate a full \[a,2\] tensor by default (for example, pypto.expand\_clone\(input, \[a,2\]\)).

    However, when the total size \[a,5\] is not divisible by the tile size \[a,2\], the valid shape (for example, \[a,1\]) of the tail tile cannot be automatically inferred by the framework. For example, the last column may contain only one element instead of a complete [a,2] tile. In this case, you must explicitly specify the actual valid shape of the tail tile through valid\_shape as follows:

    pypto.expand\_clone\(input, \[a,2\], valid\_shape = \[a, pypto.min\(2, 5 - 2 \* b\_idx\),\)

    where b\_idx indicates the loop index.

4. The dimensions of TileShape are the same as those of result, which is used to tile result.
5. There are no additional constraints on the size and shape of TileShape, except that it must not exceed the UB size.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

For example, if the input shape is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

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

Result example:

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
