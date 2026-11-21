# TileShape and Tensor Dimension Mismatch

<!-- md-trans-meta sourceCommit=8c82e872e2a7394ecc5dc0acffb03abd3567e4a3 translatedAt=2026-11-15T01:42:36.822Z pushedAt=2026-11-18T06:03:01.312Z nl=2 -->

## Symptoms

The following error is reported during operator execution:

```text
2025-12-18 10:33:06.107 E | [ExpandFunction][Function][ERROR]: Function[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0] ExpandFunction failed: Tile shape size 1 is not matched the output shape size 2.
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
