# pypto.set\_matrix\_size

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:34:01.767Z pushedAt=2026-10-28T11:57:29.601Z nl=2 -->

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

When an NZ-format input Tensor is reshaped and then used in matmul/scaled_mm for computation, the m, k, and n dimension values of the Tensor before reshaping must be passed in so that matmul/scaled_mm can obtain the original m, k, and n dimension shape values.

## Prototype

```python
set_matrix_size(size: List[int])-> None
```

## Parameters

| Parameter  | Input/Output | Description                            |
|------------|--------------|----------------------------------------|
| size       | Input        | Shape values of the m, k, and n dimensions of the input Tensor. |

## Return Value

void

## Constraints

1. When an NZ-format tensor is reshaped and then used in **matmul**/**scaled_mm** computation, this parameter must be set.

2. When the inputs to **matmul**/**scaled_mm** are 3D/4D NZ-format tensors, this parameter must be set.

## Examples

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) # Shape values of the m, k, and n dimensions of the corresponding input Tensor.
out = pypto.matmul(a, b, pypto.DT_FP32)
```
