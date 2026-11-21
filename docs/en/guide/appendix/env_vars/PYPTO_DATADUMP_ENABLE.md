# PYPTO_DATADUMP_ENABLE

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:28:31.773Z pushedAt=2026-11-18T06:03:01.280Z nl=2 -->

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
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="9030" id4 -->
- Kirin 9030 processor products: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processor products: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processor products: Not supported
<!-- end id6 -->
