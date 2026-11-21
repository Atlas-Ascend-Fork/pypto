# pypto.atomic\_add

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

Adds the input tensor src to the corresponding region of the output tensor dst in atomic mode based on the dst index position specified by offsets. Taking two dimensions as an example, the computation formula is as follows:

$$
dst\left[ offsets\left[0\right] : offsets\left[0\right] + src.shape\left[0\right],\ offsets\left[1\right] : offsets\left[1\right] + src.shape\left[1\right] \right]\ += src
$$

The same applies to other dimensions.

## Prototype

```python
atomic_add(src: Tensor, offsets: List[Union[int, SymbolicScalar]], dst: Tensor) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| src     | Input     | Source operand of the Tensor type.|
| offsets | Input     | Start offset of each dimension when src is written into dst. The element type is int or SymbolicScalar.|
| dst     | Output     | Destination operand of the Tensor type.|

## Return Value

No value is returned. The result is directly written into dst (inplace operation).

## Constraints

1. Ensure that the valid shape of the output tensor dst is correct before calling the atomic_add operation, as this API does not perform automatic shape inference.

2. The length of offsets must equal the number of dimensions of src/dst, which means $len(offsets) == src.dim == dst.dim$; otherwise, a compilation error will occur.

3. Before calling atomic_add, ensure that the data in the corresponding region of dst is valid; otherwise, undefined behavior may occur.

4. To ensure that the written region does not go out of bounds, for any dimension $i$, the condition $offsets[i] + src.shape[i] \le dst.shape[i]$ must be met; otherwise, undefined behavior occurs.

5. When multiple cores concurrently perform the atomic_add operation on overlapping regions of dst, the order in which the additions are executed is undefined.

## Examples

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
pypto.atomic_add(x, [0, 0], out)
```

Result example:

```txt
Input data x: [[1, 1],
           [1, 1]]
Output data out (before operation): [[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
Output data out (after operation): [[2, 2, 0, 0],
                      [2, 2, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]
```
