# pypto.from\_torch

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:51:10.656Z pushedAt=2026-10-28T11:57:30.172Z nl=2 -->

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

Converts a **torch.Tensor** to a **pypto.Tensor**. You can explicitly specify the name of the converted **pypto.Tensor**. You can mark specified dimensions of the converted **pypto.Tensor** as dynamic dimensions to indicate that these dimensions are variable during subsequent compilation/runtime phases.

## Prototype

```python
from_torch(tensor: torch.Tensor, name: str="", *, dynamic_axis: Optional[List[int]] = None,
           tensor_format: Optional[TileOpFormat] = None, dtype: Optional[DataType] = None) -> pypto.Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|----------------|--------------|-------------|
| tensor | Input | torch.Tensor object to be converted to pypto.Tensor. |
| name | Input | Name of the pypto.Tensor. Defaults to an empty string, indicating that from_torch automatically names it. |
| dynamic_axis | Input | List of dimension indices to be marked as dynamic. Defaults to None, indicating that no dimension is marked. |
| tensor_format | Input | pypto.TileOpFormat format to be specified. When None, it is automatically inferred based on the Tensor NPU Format. |
| dtype | Input | pypto.DataType type to be specified. When None, it is automatically inferred based on the dtype of the torch.Tensor. |

## Return Value

Returns the converted **pypto.Tensor**.

## Constraints

- The input **tensor** must be of type **torch.Tensor** or a subclass thereof.
- The input **tensor** is contiguous in the specified memory format order (`tensor.is_contiguous() == True`).
- The input **tensor** supports the following data types (`dtype`):
    - torch.float16
    - torch.bfloat16
    - torch.float32
    - torch.float64
    - torch.int8
    - torch.uint8
    - torch.int16
    - torch.uint16
    - torch.int32
    - torch.uint32
    - torch.int64
    - torch.uint64
    - torch.bool

## Examples

```python
x= torch.randn(2, 3)
x_pto = pypto.from_torch(x)
print(x_pto.shape)
y = torch.randn(2, 3)
y_pto = pypto.from_torch(y, "y", dynamic_axis=[0])
print(y_pto.shape)
z = torch.randn(2, 3)
z_pto = pypto.from_torch(z, "z", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
print(z_pto.format)
k = torch.randn(2, 3)
k_pto = pypto.from_torch(k, "k", dtype=pypto.DataType.DT_HF8)
print(k_pto.dtype)
```

The results are as follows:

```python
[2, 3]
[SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_input_tensor,0)), 3]
TileOpFormat.TILEOP_NZ
DataType.DT_HF8
```
