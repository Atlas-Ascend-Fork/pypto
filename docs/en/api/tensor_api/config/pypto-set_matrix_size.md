# pypto.set\_matrix\_size

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

When the input tensor in the NZ format is reshaped and then passed to matmul/scaled_mm for computation, the m, k, and n dimension values of the tensor before reshaping must be passed to matmul/scaled_mm, so that matmul/scaled_mm can obtain the original m, k, and n dimension values.

## Prototype

```python
set_matrix_size(size: List[int])-> None
```

## Parameters

| Parameter | Input/Output| Description                           |
|---------|-----------|-------------------------------|
| size    | Input     | Shape values of the m, k, and n dimensions of the input tensor.|

## Return Value

void

## Constraints

1. This parameter needs to be set when the input tensor in the NZ format is reshaped and then passed to matmul/scaled_mm for computation.

2. This parameter needs to be set when the input of matmul/scaled_mm is a 3D or 4D tensor in the NZ format.

## Examples

```python
a = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
pypto.set_matrix_size([32, 64, 16]) # Shape values of the m, k, and n dimensions of the input tensor.
out = pypto.matmul(a, b, pypto.DT_FP32)
```
