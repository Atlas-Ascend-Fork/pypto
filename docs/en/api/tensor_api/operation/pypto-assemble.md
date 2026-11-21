# pypto.assemble

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
<!-- npu="9050" id12 -->
- Kirin 9050 processors: Supported
<!-- end id12 -->

## Description

Assigns the input tensor input to the corresponding region of the output tensor out based on the out index position specified by offsets.

## Prototype

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor, parallel: bool = False) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| input   | Input     | Source operand.<br>The supported data types are those supported by PyPTO. <!-- npu="x90,9030,9050" id9 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id9 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| inputs   | Input     | Tuple list consisting of the source operands and output offsets.<br>Supported data types for each element are those supported by PyPTO. <!-- npu="x90,9030,9050" id10 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id10 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| offsets | Input     | Offset relative to the target output.<br>Ensure that the offsets are smaller than the shape of out.         |
| out     | Output     | Destination operand. It must have the same number of dimensions as input.<br>The supported data types are those supported by PyPTO. <!-- npu="x90,9030,9050" id11 -->For details about the supported data types of Kirin, see [Tensor Data Types](#constraints).<!-- end id11 --><br>Empty tensors are not supported. The shape size must not exceed 2147483647 (INT32_MAX).|
| parallel | Input     | Whether to allow parallel writeback. The default value is False. When the writeback regions do not overlap and can be processed safely in parallel, `parallel=True` must be explicitly passed.|

## Return Value

No value is returned. out is directly modified.

## Constraints

1. Ensure that the valid shape of the output tensor out is correct before calling the assemble operation, as this API does not perform automatic shape inference.
2. The input tensor input and the output tensor out must have the same number of dimensions.
3. When multiple assemble operations have write-after-write dependencies on overlapping regions of the same output tensor, and these writeback operations are distributed across different loop iterations or different functions, `parallel=False` ensures that the write-back operations are executed serially in dependency order. The framework marks `NORMAL` on the corresponding outcast for subsequent scheduling to process as serial writes. If the writeback regions do not overlap and can be safely processed in parallel, `parallel=True` must be explicitly passed.
<!-- npu="x90,9030,9050" id6 -->
4. A tensor supports the following data types:
   <!-- npu="9030" id7 -->
   - Kirin 9030 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id7 -->
   <!-- npu="x90" id8 -->
   - Kirin X90 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id8 -->
   <!-- npu="9050" id13 -->
   - Kirin 9050 processors: DT_FP16, DT_FP32, DT_INT32, DT_INT16, and DT_INT8
   <!-- end id13 -->
<!-- end id6 -->

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out)

y = pypto.tensor([2, 2], pypto.DT_FP32)
pypto.assemble([(x, offsets), (y, [2, 2])], out)
```

Result example:

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

### Serial Writeback Across Loops/Functions

When the same output has a write-after-write dependency across different loop iterations or functions and must be assembled serially, the default behavior is serial writeback. You can also explicitly pass `parallel=False`:

```python
# Across loops: The later write depends on the result of the earlier write. The default mode is serial.
for i in pypto.loop(0, n, name="SEQ_WRITE"):
    tile = ...
    pypto.assemble(tile, [i * tile_m, 0], out)

# Across functions: When the downstream function continues to write the overlapping region of the same out, the default mode is also serial.
pypto.assemble(partial, [offset_m, offset_n], out)
```

Note:

- For single-tensor and batch multi-source assemble, the default value is False when `parallel` is not passed. If parallel writeback is required, explicitly pass `parallel=True`.
- If the writeback regions of each assemble do not overlap and need to be scheduled in parallel, pass `parallel=True`.
