# pypto.experimental.gather\_in\_ub

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:37:49.617Z pushedAt=2026-10-28T11:57:29.902Z nl=2 -->

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

This operator supports the sparse attention mechanism. It loads the KV cache of the selected tokens from Global Memory (GM) to Unified Buffer (UB), and supports Page Attention.

## Prototype

```python
gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor,
                 block_size: int, axis: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|--------------|-----------|----------------------------------------------------------------------|
| param        | Input      | Source Operand.<br>Supported data types vary by product. For details, see [Constraints](#constraints).<br>Empty tensors are not supported. Two dimensions are supported.<br>In practice, it represents the kv cache, with a shape of [token_size, hidden_dim]. |
| indices      | Input      | Source operand.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>In practice, it represents the top k output result, whose shape is [1, k]. |
| block_table  | Input      | Source operand.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two dimensions are supported.<br>In practice, it represents the page table in Page Attention, with a shape of [1,block_table_size], where **block_table_size** is the length of the page table. |
| block_size   | Input      | Source operand.<br>Type: int.<br>Number of tokens that one block can hold in Page Attention. |
| axis         | Input      | Source operand.<br>Type: int.<br>Only the -2 axis is supported. |

## Return Value

Returns the output tensor. The data type of the tensor is the same as that of **param**, and its shape is \[k, hidden\_dim\], that is, the KV cache of the selected tokens.

## Constraints

1. This API is a custom API, and its stability is not guaranteed.
2. **param**, **indices**, and **block_table** do not support empty tensors and support only two-dimensional tensors.
3. The shape of **indices** must be [1, k].
4. The **axis** parameter supports only -2.
5. Tensor data type description:
   <!-- npu="950" id4 -->
   - Ascend 950PR&950DT products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, DT_BF16, DT_BOOL, DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 products: DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FP16, DT_FP32, and DT_BF16
   <!-- end id6 -->

## Examples

### TileShape Setting Example

Before calling this operation API, set the TileShape through **set_vec_tile_shapes**.

The dimensions of TileShape must be consistent with those of the output tensor, and are used to control the size of the output Tile block.

Take the input **$ param[token_size,hidden_dim]$** , the index **$indices[1,k]$** , the axis **$\text{axis}=-2$** , and the output **$output[k,hidden_dim]$**  as an example:

Set the TileShape to **$[k_1, hidden_dim_1]$**. This configuration directly applies to each dimension of the output, and is mapped to the input and the index. Here, **$k_1$** tiles the k dimension of indices, and **$ hidden_dim_1$** tiles the feature dimension **$hidden_dim$** of **param**. The tile memory usage must satisfy the constraint **$b_1 \cdot k_1 \cdot hidden_dim_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UB_Size}$**.

### API Call Example

![](../figures/gather_in_ub.png)

Consider the preceding scenario, where **indices** is the top k result, **block\_table** is the page table of Page Attention, **param** is the KV cache, and **block\_size** is **2**. The final result is the collection of the KV cache of the tokens.

Take token id 4 as an example (marked in red in the figure). The actual offset is computed based on block_size:

blockIdx = 4 / 2; // Compute the corresponding logical block, which is the second logical block.

tail = 4 % 2;        // Compute the intra-block offset, which is 0.

slcBlockIdx = block_table\[0, blockIdxInBatch\];  // Look up the table to obtain the actual offset of this block, which corresponds to the first physical block.

offsets = slcBlockIdx \* block_size + tail;// Compute the actual offset, which is 2.

Move the data.

```python
param = pypto.tensor([6, 4], pypto.DT_FP32)
indices = pypto.tensor([1, 3], pypto.DT_INT32)
block_table = pypto.tensor([1, 3], pypto.DT_INT32)
block_size = 2
axis = -2
result = pypto.experimental.gather_in_ub(param , indices , block_table, block_size , axis)
```

The results are as follows:

```python
Input data param:
[
  # token 0
  [  0,  1,  2,  3],
  # token 1
  [ 10, 11, 12, 13],
  # token 2
  [ 20, 21, 22, 23],
  # token 3
  [ 30, 31, 32, 33],
  # token 4
  [ 40, 41, 42, 43],
  # token 5
  [ 50, 51, 52, 53],
]
Input data indices: [0, 4, 3]
Input data block_table: [0, 2, 1]
Output data out:
[
   [  0,  1,  2,  3],
   [ 20, 21, 22, 23],
   [ 50, 51, 52, 53],
]
```
