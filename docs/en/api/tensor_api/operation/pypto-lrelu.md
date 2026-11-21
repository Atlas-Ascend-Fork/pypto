# pypto.lrelu

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

Applies the Leaky ReLU (leaky rectified linear unit) activation function to each element of the input tensor. The computation formula is as follows:

$$
\text{res}_i =
\begin{cases}
\text{input}_i & \text{if } \text{input}_i \geq 0 \\
\text{negative\_slope} \cdot \text{input}_i & \text{if } \text{input}_i < 0
\end{cases}
$$

where `negative_slope` is the negative slope parameter. The default value is `0.01`.

## Prototype

```python
lrelu(input: Tensor, negative_slope: Union[float, Element] = 0.01) -> Tensor
```

## Parameters

| Parameter        | Input/Output| Description                                                                |
|----------------|-----------|----------------------------------------------------------------------|
| input          | Input     | Source operand.<br>Supported type: Tensor.<br>Supported data types of a tensor: DT_FP16, DT_BF16, and DT_FP32.<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| negative_slope | Input     | Slope coefficient of the negative interval.<br>Supported types: float and Element. The default value is `0.01` (float type). When the type is float, it is automatically converted to the Element type, where float corresponds to DT_FP32. To use other data types, construct the value through Element.<br>The value must be a non-negative real number (≥ 0). Special values such as `nan` and `inf` are not supported.|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. The input data type must be DT_FP16, DT_BF16, or DT_FP32.
2. negative_slope must be a non-negative floating point number (≥ 0) and must not be `nan` or `inf`.
3. In-place operations are not supported (this means the output cannot share memory with the input).
4. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Note: Before calling this operation API, set TileShape through set_vec_tile_shapes.

TileShape must have the same dimensions as the output.

Example 1: If the input shape is [m, n], the output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to tile the m and n axes, respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### API Call Example

The following is the API call snippet in the computational graph construction context. The first parameter of pypto.tensor is Shape, which does not initialize the example value. The specific input data is provided by the caller during execution.

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.lrelu(a)
```

When the caller provides the following input data, the result is as follows:

```python
Input data a:   [[-1.0  0.0  1.0]]
Output data out: [[-0.01  0.0   1.0]]
```
