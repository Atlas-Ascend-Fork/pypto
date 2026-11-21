# pypto.Tensor Constructor

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
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

Creates a tensor object. When a tensor is created, it contains uninitialized random values.

## Prototype

```python
__init__(self,
         shape=None,
         dtype: Union[DataType, None] = None,
         name: str = "",
         format: TileOpFormat = TileOpFormat.TILEOP_ND,
         data_ptr: Optional[int] = None,
         device=None,
         ori_shape=None
)
```

## Parameters

| Parameter    | Input/Output| Description                                                                |
|------------|-----------|----------------------------------------------------------------------|
| shape      | Input     | Tensor shape. The value can be:<br> - None: Creates an empty tensor.<br> - List[int]: A list of integers, specifying the size of each dimension.<br> - List[Union[int, SymbolicScalar]]: A list of integers or symbolic scalars, used for dynamic shapes.|
| dtype      | Input     | Data type of the tensor.|
| name       | Input     | Tensor name.|
| format     | Input     | Tensor format. The options are as follows:<br> - TileOpFormat.TILEOP_ND (default)<br> - TileOpFormat.TILEOP_NZ |
| data_ptr   | Input     | Data pointer. The default value is None. Currently, this parameter is only used by the internal frontend framework and can be ignored by operator developers.|
| device     | Input     | Device information. The default value is None.|
| ori_shape  | Input     | Original shape, which is used to store the original shape information of a tensor. The default value is None.|

## Return Value

Tensor object.

## Constraints

None

## Examples

```python
# Create an empty tensor.
empty_tensor = pypto.Tensor()

# Create a tensor with the specified shape and data type.
tensor1 = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32)
tensor2 = pypto.Tensor(shape=[8, 16, 32], dtype=pypto.DT_INT32)

# Create a tensor with a name.
named_tensor = pypto.Tensor(shape=(4, 4),
                            dtype=pypto.DT_FP32,
                            name="input_tensor" )

# Create a tensor with a specified format.
sparse_tensor = pypto.Tensor(shape=(4, 32),
                             dtype=pypto.DT_FP32,
                             format=pypto.TileOpFormat.TILEOP_NZ )

# Create a tensor with a dynamic shape (using symbolic scalars).
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8]
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,
                              dtype=pypto.DT_FP32 )

# (Recommended) Create a tensor using the pypto.tensor convenience function.
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32)
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```
