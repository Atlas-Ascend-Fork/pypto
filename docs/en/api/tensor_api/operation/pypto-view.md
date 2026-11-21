# pypto.view

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id10 -->
- Kirin 9050 processors: Supported
<!-- end id10 -->

## Description

Extracts a partial view from the input tensor for subsequent computation.

## Prototype

```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor
```

## Parameters

| Parameter     | Input/Output| Description                                                                |
|-------------|-----------|----------------------------------------------------------------------|
| input       | Input     | Source operand.<br>The supported data types are those supported by PyPTO. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| shape       | Input     | Size of the view to be extracted, which must have the same number of dimensions as input.<br>The shape size must be less than or equal to 2147483647 (INT32_MAX). **The shape supports only the List [int] type and does not support the SymbolicScalar type.**|
| offsets     | Input     | Offset of each dimension relative to input when extracting the view.<br>offsets must be smaller than the shape of the input.|
| valid_shape | Input     | Valid data size of the extracted view block.<br>valid_shape must be smaller than the shape of input. In scenarios such as page_attention, when the input tensors such as kv_cache contain invalid data, the output valid shape cannot be correctly inferred and must be passed manually.|
| dtype       | Input     | Data type of the return value, which allows the input data to be interpreted as a different data type.|

`valid_shape` can be used together with `pypto.experimental.assume_divisible`. When a dynamic scalar is known to be exactly divisible by TileShape, you can use `assume_divisible` to declare this fact, helping the compiler eliminate the per-tile dynamic valid shape along that axis and enabling optimizations that depend on the static valid shape, such as dualdst.

## Return Value

Returns the output tensor. Its data type is the same as that of input, and its shape is the size specified by shape. If valid\_shape is specified, the actual size is valid\_shape. If dtype is specified, the input is read according to dtype.

## Constraints

- **valid_shape requires pypto.view**: When `valid_shape` (dynamic valid data size) needs to be specified, the `[]` slicing syntax cannot be used. Instead, the explicit `pypto.view` API must be used.
- The input tensor input and the input shape must have the same number of dimensions.
- **A view is independent of the source input after creation**: Once created, a view becomes an independent data copy. Reads and writes to the view (including view[:] = ...) affect only the view itself; they do not write back to the source input, nor do they observe subsequent modifications to the source input. To write data back to the source tensor (for example, when accumulating writes to a persistent buffer slice by slice in a loop), use pypto.assemble(value, offsets, dest).
<!-- npu="x90,9030,9050" id6 -->
- A tensor supports the following data types:
   <!-- npu="9030" id7 -->
   - Kirin 9030 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id7 -->
   <!-- npu="x90" id8 -->
   - Kirin X90 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id8 -->
   <!-- npu="9050" id11 -->
   - Kirin 9050 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id11 -->
<!-- end id6 -->

## Examples

- Basic usage

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)
    ```

    Result example:

    ```python
    Input data x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4]]
    Output data y: [[3 3 4 4],
                [3 3 4 4],
                [3 3 4 4],
                [3 3 4 4]]
    ```

- Adding valid\_shape

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape)
    ```

    Result example:

    ```python
    Input data x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 5 5 6 6],
                [1 1 2 2 5 5 6 6]]
    Output data y: [[5 5 6 6],
                [5 5 6 6],
                [0 0 0 0],
                [0 0 0 0]]
    ```

- Specifying dtype

    ```python
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    ```

    Result example:

    ```python
    Input data x:
    [[0.9405094  0.20237109],
     [0.99819463 0.13246714]]

    Output data y:
    [[  57  -59  112   63   94   58   79   62],
     [ -81 -119  127   63  119  -91    7   62]]

    ```
