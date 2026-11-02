# pypto.Tensor Constructor

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:59:21.354Z pushedAt=2026-10-28T11:57:30.234Z nl=2 -->

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

Creates a tensor object. The tensor contains uninitialized random values upon creation.

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

| Parameter | Input/Output | Description                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| shape      | Input      | Shape of the tensor, which can be one of the following types:<br> - None: Creates an empty tensor.<br> - List[int]: A list of integers specifying the size of each dimension.<br> - List[Union[int, SymbolicScalar]]: A list containing integers or symbolic scalars, used for dynamic shapes. |
| dtype      | Input      | Data type of the tensor. |
| name       | Input      | Name of the tensor. |
| format     | Input      | Format of the tensor. Available values include:<br> - TileOpFormat.TILEOP_ND (default)<br> - TileOpFormat.TILEOP_NZ |
| data_ptr   | Input      | Data pointer, defaults to None. Currently used only internally by the frontend framework; operator developers can ignore it. |
| device     | Input      | Device information, defaults to None. |
| ori_shape  | Input      | Original shape, used to store the original shape information of the tensor, defaults to None. |

## Return Value

A tensor object.

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

# Create a tensor with the specified format.
sparse_tensor = pypto.Tensor(shape=(4, 32),
                             dtype=pypto.DT_FP32,
                             format=pypto.TileOpFormat.TILEOP_NZ )

# Create a tensor with a dynamic shape (using symbolic scalars).
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8]
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,
                              dtype=pypto.DT_FP32 )

# Create a tensor using the pypto.tensor convenience function (recommended).
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32)
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```
