# TileShape and Tensor Dimension Mismatch

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:20:45.395Z pushedAt=2026-10-27T02:45:13.933Z nl=2 -->

## Symptoms

The following error is reported during operator execution:

```text
2025-12-18 10:33:06.107 E | [ExpandFunction][Function][ERROR]: FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0] ExpandFunction failed: Tile shape size 1 is not matched the output shape size 2.
2025-12-18 10:33:06.107 E | Run pass [ExpandFunction] failed.
2025-12-18 10:33:06.107 E | Run pass <ExpandFunction> failed
```

## Possible Causes

The TileShape of an operation is set with a dimension that is too small, smaller than the shape dimension of the output tensor of the operation, causing the error.

## Solution

Locate the corresponding loop based on the error message. As described below, the problematic code appears in the `b_loop` loop:

```text
FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0]
```

After locating the corresponding loop, check the error dimension in the log and review the code logic to confirm that TileShape has dimension 1 while the output shape has dimension 2. Then reset TileShape to 2 dimensions.
