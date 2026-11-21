# pypto.index\_select

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

Returns a new tensor that indexes the input tensor along dimension `dim` using the elements in `index`.

The returned tensor has the same number of dimensions as the original tensor (input). The size of the `dim` dimension has the same length as that defined by `index`. The sizes of other dimensions are the same as those of the original tensor.

$$
\begin{array}{l}
\text{shape}(\mathbf{input}) = (S_0, S_1, \ldots, S_{n-1}) \\
dim = d \\
\text{shape}(\mathbf{index}) = (I_0,) \\
\text{shape}(\mathbf{result}) = (S_0, \ldots, S_{d-1}, I_0, S_{d+1}, \ldots, S_{n-1}) \\
\mathbf{result}[s_0, \ldots, s_{d-1}, i, s_{d+1}, \ldots, s_{n-1}] = \mathbf{input}[s_0, \ldots, s_{d-1}, \mathbf{index}[i], s_{d+1}, \ldots, s_{n-1}]
\end{array}
$$

## Prototype

```python
index_select(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>Supported type: Tensor. The supported data types vary by model. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. The shape supports only 1 to 4 dimensions. The shape size must not exceed 2147483647 (INT32_MAX).|
| dim     | Input     | Dimension to index, with the int type.<br>It supports any value not exceeding the number of dimensions of input. For details, see the "Constraints" section.|
| index   | Input     | Source operand.<br>Supported type: Tensor.<br>The tensor supports the following data types: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. The shape supports only 1 to 2 dimensions. The shape size must not exceed 2147483647 (INT32_MAX) and the values must be valid indices, meaning not exceeding the shape size of the input along the dim axis.

## Return Value

Returns the output tensor. Its data type is the same as that of the input. Its shape is determined by input, dim, and index. For details, see the Description section.

## Constraints

1. index must be of the integer type (DT\_INT32 or DT\_INT64), and its values must be valid indices, meaning not exceeding input.shape[dim].

2. dim must be of the int type, with a value range of -input.dim <= dim < input.dim. Negative numbers are supported and are interpreted as dim + input.dim.

3. viewshape of input.shape along the dim axis cannot be tiled, requiring viewshape\[dim\] \>= input.shape\[dim\]. The shape sizes of the remaining dimensions are not restricted. This constraint comes from the operator semantics of index_select: the dim axis serves as the index source and must be fully visible in the current view, rather than being an additional constraint of the current implementation. If the dim axis is tiled with a viewshape smaller than input.shape\[dim\], the index may reference data outside the current view, resulting in precision errors or an AI Core error.

4. A tensor supports the following data types:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, DT_BF16, DT_BOOL, DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->
5. The dimensions of TileShape are the same as those of result, which is used to tile result. When setting TileShape, ensure that the size of result does not exceed the UB size. For details, see [TileShape Setting Example](#tileshape-setting-example).

6. Tensor inputs do not support the `TileOpFormat.TILEOP_NZ` format.

## Examples

### TileShape Setting Example

Before calling this operation API, set TileShape through set_vec_tile_shapes.

The dimensions of TileShape must be consistent with those of the output tensor, and are used to control the size of the output Tile block.

Take input $input[B,S,D]$, index $index[T]$, axis $\text{axis}=-2$, and output $output[B,T,D]$ as an example. If TileShape is set to $[b_1, t_1, d_1]$, the configuration directly applies to each dimension of the output and is mapped to the input and index. Here, $b_1$ tiles the batch dimension B of the input, and $d_1$ tiles the feature dimension D of the input. The input sequence dimension S (axis-2) is not tiled and serves only as the index source, requiring viewshape to cover the complete S axis. $t_1$ applies to the length dimension T of the index. The tile memory usage must satisfy $b_1 \cdot t_1 \cdot d_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UBSize}$.

### API Call Example

```python
x = pypto.tensor([3, 4], pypto.DT_FP32)
indices = pypto.tensor([2], pypto.DT_INT32)
out1 = pypto.index_select(x, 0, indices)
out2 = pypto.index_select(x, 1, indices)
```

Result example:

```python
Input x:        [[ 0.1427,  0.0231, -0.5414, -1.0009],
               [-0.4664,  0.2647, -0.1228, -1.1068],
               [-1.1734, -0.6571,  0.7230, -0.6004]]
Input index:    [0, 2]
Output out1:     [[ 0.1427,  0.0231, -0.5414, -1.0009],
               [-1.1734, -0.6571,  0.7230, -0.6004]]
Output out2:     [[ 0.1427, -0.5414],
               [-0.4664, -0.1228],
               [-1.1734,  0.7230]]
```
