# pypto.experimental.gather\_in\_l1

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

This API is a custom API and its stability is not guaranteed.

Discretely copies the data of specified rows from a tensor in GM to L1 Buffer, copying the first size elements of each row.

## Prototype

```python
gather_in_l1(src: Tensor, indices: Tensor, block_table: Tensor, block_size: int,
                 size: int, is_b_matrix: bool, is_trans: bool) -> Tensor
```

## Parameters

| Parameter      | Input/Output| Description                                                                |
|--------------|-----------|----------------------------------------------------------------------|
| src          | Input     | Source operand.<br>Supported data types: DT_FP32, DT_FP16, DT_BF16, and DT_INT8.<br>Empty tensors are not supported. Two dimensions are supported.|
| indices      | Input     | Row offset of the source operand.<br>Supported data types: DT_INT32 and DT_INT64.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>The shape is [1, n].|
| block_table  | Input     | Page table.<br>Supported data type: DT_INT32.<br>Empty tensors are not supported. Two-dimensional tensors are supported.<br>In practice, it represents the page table in Page Attention, with a shape of [1,block_table_size], where block_table_size indicates the length of the page table.|
| block_size   | Input     | Source operand.<br>Type: int.<br>Number of tokens that one block can hold in Page Attention.|
| size         | Input     | Number of elements copied per row.<br>The number of elements must be less than the number of columns of the source operand.|
| is_b_matrix  | Input     | Whether the copied result (output tensor) is used as the B matrix of matmul.|
| is_trans     | Input     | Whether the copied result (output tensor) is transposed.|

## Return Value

Return an output tensor.

## Constraints

This API is for trial use and may be changed in later versions. It is not available in production environments.

## Examples

```python
src = pypto.tensor([16, 32], pypto.DT_FP32, "tensor_src")
offset = pypto.tensor([1, 32], pypto.DT_INT32, "tensor_offset")
block_table = pypto.tensor([1, 4], pypto.DT_INT32, "block_table")
block_size = 2
size = 16
is_b_matrix = False
is_trans = False
out = pypto.experimental.gather_in_l1(src, offset, block_table, block_size, size, is_b_matrix, is_trans)
```
