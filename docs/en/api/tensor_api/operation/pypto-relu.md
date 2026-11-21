# pypto.relu

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Not supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Not supported
<!-- end id8 -->
<!-- npu="9050" id9 -->
- Kirin 9050 processors: Not supported
<!-- end id9 -->

## Description

Performs the rectified linear unit (ReLU) operation on each element of the input, retaining only the positive part and setting negative values to 0. The computation formula is as follows:

$$
res_i = \max(0, input_i)
$$

## Prototype

```python
relu(input: Tensor) -> Tensor
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| input  | Input     | Source operand.<br>Supported type: Tensor.<br>The data types supported by tensors vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|

## Return Value

Returns the output tensor. Its data type and shape are the same as those of input.

## Constraints

1. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT16, DT_INT32, DT_INT64, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT16, DT_INT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT16, DT_INT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->
2. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

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
input_tensor = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.relu(input_tensor)
```

When the caller provides the following input data, the result is as follows:

```python
Input data input: [[-2.0 0.0 3.0]]
Output data out: [[ 0.0 0.0 3.0]]
```
