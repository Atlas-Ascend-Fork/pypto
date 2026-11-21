# PYPTO_PROF_PMU_EVENT_TYPE

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:29:57.902Z pushedAt=2026-11-18T06:03:01.284Z nl=2 -->

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
