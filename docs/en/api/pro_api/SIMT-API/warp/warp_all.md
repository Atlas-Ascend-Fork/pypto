# pypto_pro.language.simt.warp_all

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

Determines whether the inputs of all active lanes in the current warp are true. This can be understood as performing a logical AND operation on the predicates of all active lanes.

Each active lane independently computes its own predicate and then participates in the vote:

- If the predicates of all active lanes are true, the vote result is 1.
- If the predicate of any active lane is false, the vote result is 0.
- Each lane participating in the vote obtains the same vote result. Inactive lanes do not participate in the vote.

## Prototype

```python
pypto_pro.language.simt.warp_all(
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

Returns the vote result of the DT_INT32 type. If the predicates of all active lanes are true, 1 is returned. Otherwise, 0 is returned. All lanes participating in the vote return the same result.

## Examples

The following example performs two votes. In the first vote, the conditions of all lanes are true, and all values in row 0 of the output are 1. In the second vote, the condition of lane 31 is false, and all values in row 1 of the output are 0.

```python
import pypto_pro.language as pl

WARP_SIZE = 32


@pl.vector_function(mode="simt", max_threads=WARP_SIZE)
def warp_all_example(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    tid = pl.simt.linear_thread_idx()
    lane = pl.simt.lane_id()
    warp_size = pl.simt.warp_size()
    output[0, tid] = pl.simt.warp_all(lane < warp_size)
    output[1, tid] = pl.simt.warp_all(lane < warp_size - 1)


@pl.jit()
def warp_all_kernel(output: pl.Tensor[[2, WARP_SIZE], pl.DT_INT32]):
    with pl.section_vector():
        warp_all_example[WARP_SIZE](output)
```
