# PYPTO_DATADUMP_ENABLE

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T01:11:18.685Z pushedAt=2026-10-27T02:32:50.341Z nl=2 -->

## Description

Enables tensor dumping during on-device execution. When enabled, the framework automatically dumps the input and output data of leaf functions during NPU operator execution, for precision comparison against simulation results.

- Type: string
- Value range: `true` (enabled), `false`, or unset (disabled)

## Configuration Example

```bash
# Method 1: Set in the command line
export PYPTO_DATADUMP_ENABLE=true

# Method 2: Set in Python code
import os
os.environ["PYPTO_DATADUMP_ENABLE"] = "true"
```

## Constraints

- Must be used together with `enable_pass_verify` and `pass_verify_save_tensor` in `verify_options`.
- Dump data is written to the `output/output_*/tensor/` directory.
- Takes effect only during on-device execution on the NPU, and is invalid in simulation mode.

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
