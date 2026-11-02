# PYPTO_PROF_PMU_EVENT_TYPE

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:12:35.859Z pushedAt=2026-10-27T02:33:39.495Z nl=2 -->

## Description

Selects the PMU (Performance Monitoring Unit) data collection mode. The PMU monitors hardware performance events of the AI Core, and different modes correspond to different event groups.

- Type: integer
- Value range: `1`, `2`, `4`, `5`, `6`, `7`, `8`. The default value is `2`.

## Configuration Example

```bash
export PYPTO_PROF_PMU_EVENT_TYPE=2

# Collect PMU data
msprof --task-time=l3 --output=./prof_data python xxx.py

# Parse data
python tools/profiling/tilefwk_pmu_to_csv.py -p PROF_xxx/device_x/data -pe=$PYPTO_PROF_PMU_EVENT_TYPE --arch dav_3510
```

## Constraints

- For the complete PMU collection process, see [Collecting PMU Data](../../programming_guide/tensor/debug/performance.md).

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
