# pypto.softmax

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

Applies the softmax function to the input tensor in the specified dimension and normalizes the elements in that dimension to a probability distribution with values ranging from 0 to 1 (the sum of all elements is 1). The computation formula is as follows:

$$
\text{softmax}(input)_i = \frac{e^{input_i - \text{max}(input)}}{\sum_{j=1}^{k} e^{input_j - \text{max}(input)}}
$$

## Prototype

```python
softmax(input: Tensor, dim: int) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported data type: DT_FP32. <!-- npu="x90,9030,9050" id11 -->Kirin processors: DT_FP16 and DT_FP32.<!-- end id11 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension along which normalization is performed.<br>Negative indexing is supported (for example, -1 indicates the last dimension).<br>The value must be within the range of [-input.dim, input.dim-1]. <!-- npu="x90,9030,9050" id12 -->Kirin processors: -1.<!-- end id12 --> |

## Return Value

Returns a tensor. Its shape and data type are the same as those of the input tensor, and the sum of elements along the dimension specified by dim is 1.

## Constraints

None

## Examples

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.softmax(x, -1)
```

Result example:

```python
Input data x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
Output data y: [[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
```
