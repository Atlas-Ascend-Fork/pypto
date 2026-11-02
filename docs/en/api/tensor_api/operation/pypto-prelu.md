# pypto.prelu

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:16:26.427Z pushedAt=2026-10-28T11:57:30.040Z nl=2 -->

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

Performs the parametric rectified linear unit (PReLU) operation on each element of **input**. When an element value is greater than or equal to 0, it remains unchanged; when it is less than 0, it is multiplied by a weight coefficient. The computation formula is as follows:

$$
res_i = \begin{cases}
input_i & \text{if } input_i \geq 0 \\
weight_i \times input_i & \text{if } input_i < 0
\end{cases}
$$

Here, **weight** is a one-dimensional tensor:

- When **input** is 1-dimensional, the length of **weight** is 1, and the weight is shared across elements.
- When **input** is 2- to 4-dimensional, the length of **weight** is the same as the size of the second dimension (channel dimension) of **input**, and the weight is shared by channel.

## Prototype

```python
prelu(input: Tensor, weight: Tensor) -> Tensor
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input      | Source operand.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16.<br>Empty tensors are not supported. The shape supports 1 to 4 dimensions. The shape size is not greater than 2147483647 (INT32_MAX). |
| weight  | Input      | Weight parameter.<br>Supported type: Tensor.<br>Supported data types of Tensor: DT_FP16, DT_FP32, and DT_BF16, which must be the same as the type of **input**.<br>The shape is one-dimensional. When **input** is 1-dimensional, the length is 1; when **input** is 2- to 4-dimensional, the length is the same as the size of the second dimension of **input**. |

## Return Value

Returns the output tensor, whose data type and shape are the same as those of **input**.

## Constraints

1. The types of **input** and **weight** must be the same.
2. The shape of **weight** must be one-dimensional. When **input** is 1-dimensional, the length is 1; when **input** is 2- to 4-dimensional, the length is equal to the size of the second dimension of **input**.
3. Because temporary memory is used, when the input is two-dimensional, the **TileShape** size has an additional constraint. Assuming that **TileShape** is \[a,b\], then a*b*sizeof(self) + b/8 + 8KB < UB.
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The TileShape must have the same number of dimensions as the output.

Example 1 (1D input): The input **input** and **weight** have shapes [n] and [1], respectively. The output is [n], and **TileShape** is set to [n1], where n1 is used to split the n axis.

Example 2 (2D input): The input input and weight have shapes [m, n] [n]. The output is [m, n], and TileShape is set to [m1, n1], where m1 and n1 are used to split the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

Example 1: 1D input

```python
# Example 1: 1D PReLU operation.
# The input shape is [4], and the weight shape is [1].
# For negative elements, all elements share the same weight.
input_tensor = pypto.tensor([-2.0, 1.0, -3.0, 0.5], pypto.DT_FP32)
weight_tensor = pypto.tensor([0.25], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

The results are as follows:

```python
Input data input:  [-2.0,  1.0, -3.0,  0.5]
Input data weight: [ 0.25 ]
Output data out:    [-0.5,  1.0, -0.75,  0.5]
```

Computation process description:

- All elements share the weight 0.25.
- -2.0 < 0, result = 0.25 × (-2.0) = -0.5; 1.0 ≥ 0, result = 1.0.
- -3.0 < 0, result = 0.25 × (-3.0) = -0.75; 0.5 ≥ 0, result = 0.5

Example 2: 2D Input

```python
# Example 2: 2D PReLU operation
# input shape is [2, 3], weight shape is [3]
# For negative elements, multiply by the corresponding weight per channel
input_tensor = pypto.tensor([[-2.0, 1.0, -3.0], [0.5, -1.0, 2.0]], pypto.DT_FP32)
weight_tensor = pypto.tensor([0.25, 0.5, 0.1], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

The results are as follows:

```python
Input data input:  [[-2.0,  1.0, -3.0], [ 0.5, -1.0,  2.0]]
Input data weight: [ 0.25, 0.5,  0.1]
Output data out:    [[-0.5,  1.0, -0.3], [ 0.5, -0.5,  2.0]]
```

Computation process description:

- Channel 0: -2.0 < 0, result = 0.25 × (-2.0) = -0.5; 0.5 ≥ 0, result = 0.5
- Channel 1: 1.0 ≥ 0, result = 1.0; -1.0 < 0, result = 0.5 × (-1.0) = -0.5
- Channel 2: -3.0 < 0, result = 0.1 × (-3.0) = -0.3; 2.0 ≥ 0, result = 2.0
