# pypto.experimental.transposed_batchmatmul

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:37:07.908Z pushedAt=2026-10-28T11:57:29.896Z nl=2 -->

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

This is a custom API with many constraints. Its stability is not guaranteed.

This operator performs transposed batch matrix multiplication. The specific operations are as follows:

1. Transpose the input tensor `tensor_a` from shape (M, B, K) to (B, M, K).
2. Perform batch matrix multiplication by multiplying the transposed `tensor_a` (B, M, K) with `tensor_b` (B, K, N) to obtain the intermediate result (B, M, N).
3. Transpose the intermediate result back to shape (M, B, N) as the final output.

## Prototype

```python
transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype: dtype) -> Tensor
```

## Parameters

| Parameter    | Input/Output | Description                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| tensor_a  | Input      | Left input tensor.<br>Supported data types: DT_FP16 and DT_BF16.<br>Empty tensors are not supported; three-dimensional tensors are supported.<br>The shape must be (M, B, K). |
| tensor_b  | Input      | Right input tensor.<br>Supported data types: DT_FP16 and DT_BF16.<br>Empty tensors are not supported; three-dimensional tensors are supported.<br>The shape must be (B, K, N). |
| out_dtype | Input      | Data type of the output tensor.<br>Supported data types: DT_FP16 and DT_BF16. |

## Return Value

Returns the output tensor, whose data type is specified by `out_dtype` and whose shape is (M, B, N).

## Constraints

This API is an experimental feature and may change in later versions. It is not intended for use in production environments.

## Examples

```python
import pypto

# Create the input tensor.
a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")

# Call the operator.
c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)

# The shape of the output tensor c is (16, 2, 64).
```
