# pypto.rms\_norm

<!-- md-trans-meta sourceCommit=74805bf4f8e98b142f26afa769f2f90aee900ccb translatedAt=2026-10-26T17:24:17.316Z pushedAt=2026-10-28T11:57:30.067Z nl=2 -->

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

Performs Root Mean Square LayerNorm (RMSNorm) along the last dimension. If **gamma** is provided, element-wise scaling is applied along the last dimension.

## Prototype

```python
rms_norm(input: Tensor, gamma: Tensor = None, epsilon: float = 1e-6) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported data types: data types supported by PyPTO.<br>Can be a tensor of any shape [..., C], where the last dimension C usually represents the number of channels or features. |
| gamma   | Input      | Optional scaling parameter, whose shape should be [C]. |
| epsilon | Input      | Numerical stability constant, defaulting to 1e-6. |

## Return Value

Returns the normalized tensor with the same shape as the input tensor. The output tensor is converted back to the original data type of the input tensor.

## Constraints

None

## Examples

```python
x = pypto.tensor([2, 4], pypto.DT_FP32)
gamma = pypto.tensor([4], pypto.DT_FP32)
y = pypto.rms_norm(x, gamma)
```

The results are as follows:

```python
Input data x: [[1, 2, 3, 4],
            [5, 6, 7, 8]]
Input data gamma: [1, 1, 1, 1]
Output data y: [[0.3651, 0.7302, 1.0954, 1.4605],
            [0.7580, 0.9097, 1.0613, 1.2129]]
```
