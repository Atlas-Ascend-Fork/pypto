# DUMP_DEVICE_PERF

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-27T01:11:19.901Z pushedAt=2026-10-27T02:32:38.220Z nl=2 -->

## Description

Enables joint AI CPU/AI Core performance data collection. When enabled, the framework collects end-to-end latency data during AI CPU scheduling and AI Core execution, and prints relevant performance statistics to the device. The collection results are saved to the `output/output_timestamp/` directory.

- Type: string
- Value range: `true` (enabled), `false`, or unset (disabled)

## Configuration Example

```bash
export DUMP_DEVICE_PERF=true
python3 examples/02_intermediate/operators/softmax/softmax.py

# Analyze the data after collection.
python tools/scripts/machine_perf_trace.py analyze output/output_<timestamp>/machine_trace_perf_data_0.json
```

## Constraints

- A maximum of 200 rounds of data collection and screen output are supported. The excess will be truncated.
- Currently, only data from 200 devTask constructions can be collected. A warning will appear in the log when this limit is exceeded.

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
