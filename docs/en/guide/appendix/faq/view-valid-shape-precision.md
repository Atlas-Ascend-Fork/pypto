# Precision Issue Occurs When valid_shape Is Not Passed to view

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:21:34.228Z pushedAt=2026-10-27T02:49:20.111Z nl=2 -->

## Symptoms

In some scenarios, precision issues occur when `valid_shape` is not passed to `view`.

## Possible Causes

When the input tensor of the `view` API does not have a correct valid_shape, the output valid_shape cannot be correctly inferred by the framework.

## Solution

If you suspect that the validShape inference for `view` is incorrect, first pass a `valid_shape` to `view` and check whether the output meets expectations.

A typical scenario where `valid_shape` must be passed:

When the input valid_shape depends on another tensor identifier, `valid_shape` must be passed. In the following scenario, the valid_shape `curSeq` of `q0` comes from another tensor and cannot be obtained through inference:

```python
# Input input [B, S, H]
# Input act_seqs [B]
# Output out [B, S, H]
# Computation process AddS
# The code is as follows:
for b_idx in pypto.loop(B, name="b_loop", idx_name="b"):
    cur_seq = act_seqs[b_idx]
    a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
    a1 = a0 + 1.0
    out[b_idx, :, :] = a1
```
