# pypto.assemble

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:03:17.680Z pushedAt=2026-10-28T11:57:29.761Z nl=2 -->

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

Assigns the input tensor `input` to the corresponding region of the output tensor `out`, based on the output index position specified by `offsets`.

## Prototype

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor, parallel: bool = False) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| input | Input | Source operand.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| inputs | Input | Tuple list consisting of source operands and output offsets.<br>Supported data type for each element: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| offsets | Input | Offset relative to the target output.<br>Ensure that offsets are smaller than the shape of `out`. |
| out | Output | Destination operand, which must have the same number of dimensions as `input`.<br>Supported data types: data types supported by PyPto.<br>Empty tensors are not supported. The shape size must not exceed **2147483647** (that is, **INT32_MAX**). |
| parallel | Input | Whether to allow parallel writeback. Defaults to False; when the writeback regions do not overlap and can be safely parallelized, explicitly pass `parallel=True`. |

## Return Value

No return value; **out** is modified directly.

## Constraints

1. Ensure that the valid shape of the output tensor `out` is correct before calling `assemble`, as this API does not perform automatic shape inference.
2. The input tensor `input` and the output tensor `out` must have the same number of dimensions.
3. When multiple assemble operations have write-after-write dependencies on overlapping regions of the same out, and these writebacks are distributed across different loop iterations or different functions, the default `parallel=False` already ensures that the writebacks are executed serially in dependency order. The framework marks `NORMAL` on the corresponding outcast for subsequent scheduling to process as serial writes. If the writeback regions do not overlap and can be safely parallelized, explicitly pass `parallel=True`.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out)

y = pypto.tensor([2, 2], pypto.DT_FP32)
pypto.assemble([(x, offsets), (y, [2, 2])], out)
```

The results are as follows:

```python
Output data x: [[1, 1]
           [1, 1]]
Input data out: [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
Output data out: [[1, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
Output data out1: [[1, 1, 0, 0],
              [1, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 1, 1]]
```

### Serial Writeback Across Loops / Functions

When the same output has a write-after-write dependency across different loop iterations or different functions and must be assembled serially, serial writeback is the default behavior. You can also explicitly pass `parallel=False`:

```python
# Across loops: the later write depends on the result of the earlier write, and serial execution is the default.
for i in pypto.loop(0, n, name="SEQ_WRITE"):
    tile = ...
    pypto.assemble(tile, [i * tile_m, 0], out)

# Across functions: when a downstream function continues writing to the overlapping region of the same out, serial execution is also the default.
pypto.assemble(partial, [offset_m, offset_n], out)
```

Description:

- For both single-tensor and batched multi-source assemble, the default is False when `parallel` is not passed. To enable parallel writeback, explicitly pass `parallel=True`.
- If the writeback regions of each assemble do not overlap and parallel scheduling is required, set `parallel=True`.
