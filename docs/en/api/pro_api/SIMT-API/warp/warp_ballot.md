# pypto_pro.language.simt.warp_ballot

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Determines whether the input of each active lane in the current warp is true, and summarizes the results into a 32-bit unsigned integer.

Each bit in the return value corresponds to a lane.

- The ith bit corresponds to lane i.
- If lane i is active and its predicate is true, the ith bit is 1.
- If lane i is inactive or its predicate is false, the ith bit is 0.
- Each lane participating in the vote obtains the same 32-bit bitmap.

The difference between warp_ballot, warp_all, and warp_any is that warp_all and warp_any return only the overall result, and warp_ballot retains the result and position of each lane.

## Prototype

```python
pypto_pro.language.simt.warp_ballot(
    predicate: int,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| predicate | Input| Condition value for the current lane to participate in the vote, which can be a Boolean or integer constant, or a runtime scalar expression of the corresponding type. DT_BOOL and DT_INT32 are supported. The value 0 indicates false, and a non-zero value indicates true. Different condition values can be passed for different lanes.|

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.
- Only the active lanes that have executed the current instruction are summarized. When branch divergence occurs, the participating sets of different branches are different.

## Return Value

Returns the condition bitmap of the DT_UINT32 type. Bit i is 1 only when lane i is active and its predicate is true. The corresponding bit of an inactive lane is 0. All lanes participating in the vote return the same result.

## Examples

The following example performs two votes. The bits are arranged from bit 31 to bit 0. In the first vote, all even-numbered lanes vote in favor, and the result is 01010101 01010101 01010101 01010101. In the second vote, the first 16 lanes vote in favor, and the result is 00000000 00000000 11111111 11111111. The result of each vote is written to the output positions corresponding to all lanes.

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_ballot_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    output[0, tid] = pl.simt.warp_ballot((lane % 2) == 0)
    output[1, tid] = pl.simt.warp_ballot(lane < pl.simt.warp_size() // 2)


@pl.jit()
def warp_ballot_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_UINT32]):
    with pl.section_vector():
        warp_ballot_example[WARP_SIZE](output)
```
