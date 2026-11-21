# pypto.prelu

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

Performs the parametric rectified linear unit (PReLU) operation on each element of input. When an element value is greater than or equal to 0, it remains unchanged; when it is less than 0, it is multiplied by a weight coefficient. The computation formula is as follows:

$$
res_i = \begin{cases}
input_i & \text{if } input_i \geq 0 \\
weight_i \times input_i & \text{if } input_i < 0
\end{cases}
$$

where weight is a 1D tensor:

- When input is 1D, the length of weight is 1, and the weight is shared across elements.
- When input is 2D to 4D, the length of weight is the same as the size of the second dimension (channel dimension) of input, and the weight is shared across channels.

## Prototype

```python
prelu(input: Tensor, weight: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| weight  | Input     | Weight parameter.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_FP32, and DT_BF16. The data type must be the same as that of input.<br>The shape is 1D. If input is 1D, the length is 1. If input is 2D to 4D, the length is the same as the size of the second dimension of input.|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. The input and weight types must be the same.
2. The shape of weight must be 1D. If input is 1D, the length is 1. If input is 2D to 4D, the length is the same as the size of the second dimension of input.
3. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.
4. Due to temporary memory usage, when the input is 2D, the TileShape size has additional constraints. When an input tensor is copied to the UB, the last dimension is 32-byte aligned and padded, and then the second-to-last dimension is row-padded. For an output tensor, only the last dimension is 32-byte aligned and padded. The weight is not padded. Assume that TileShape is \[a,b\], and let $A=32/sizeof(input), b_{align}=CeilAlign(b, A), k=b_{align}/A, p=\lceil8/k\rceil, and a_{pad}=a+p-1$. Then, the total UB space usage is as follows:

   $$
   (a_{pad}+a+1)*b_{align}*sizeof(input)+b/8+8\text{KB} <= UB
   $$

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1 (1D input): The input and weight have shapes [n] and [1], respectively. If the output is [n] and TileShape is set to [n1], then n1 is used to tile the n axis.

Example 2 (2D input): The input and weight have shapes [m, n] and [n], respectively. If the output is [m, n] and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

Select the corresponding TileShape based on the input dimension. For details, see [API Call Example](#api-call-example).

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

Example 1: 1D input

```python
# Example 1: 1D PReLU operation.
# The input shape is [4] and the weight shape is [1].
# For negative elements, all elements share the same weight.
pypto.set_vec_tile_shapes(16)
input_tensor = pypto.tensor([4], pypto.DT_FP32)
weight_tensor = pypto.tensor([1], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

When the caller provides the following input data, the result is as follows:

```python
Input data input:  [-2.0,  1.0, -3.0,  0.5]
Input data weight: [0.25 ]
Output data out:    [-0.5,  1.0, -0.75,  0.5]
```

Computation process description:

- All elements share weight 0.25.
- -2.0 < 0, result = 0.25 × (-2.0) = -0.5; 1.0 ≥ 0, result = 1.0.
- -3.0 < 0, result = 0.25 × (-3.0) = -0.75; 0.5 ≥ 0, result = 0.5

Example 2: 2D input

```python
# Example 2: 2D PReLU operation
# The input shape is [2, 3], and the weight shape is [3].
# Multiply each negative element by the corresponding weight per channel.
pypto.set_vec_tile_shapes(4, 16)
input_tensor = pypto.tensor([2, 3], pypto.DT_FP32)
weight_tensor = pypto.tensor([3], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

When the caller provides the following input data, the result is as follows:

```python
Input data input:  [[-2.0,  1.0, -3.0], [ 0.5, -1.0,  2.0]]
Input data weight: [ 0.25, 0.5,  0.1]
Output data out:    [[-0.5,  1.0, -0.3], [ 0.5, -0.5,  2.0]]
```

Computation process description:

- Channel 0: -2.0 < 0, result = 0.25 × (-2.0) = -0.5; 0.5 ≥ 0, result = 0.5
- Channel 1: 1.0 ≥ 0, result = 1.0; -1.0 < 0, result = 0.5 × (-1.0) = -0.5
- Channel 2: -3.0 < 0, result = 0.1 × (-3.0) = -0.3; 2.0 ≥ 0, result = 2.0
