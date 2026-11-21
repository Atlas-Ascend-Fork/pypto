# PYPTO_LAUNCH_SCHED_SAME_CLUSTER

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:29:12.955Z pushedAt=2026-11-18T06:03:01.283Z nl=2 -->

## Description

Controls whether to forcibly allocate scheduling threads (AI CPU) within the same cluster. Allocating scheduling threads within the same cluster can achieve better inter-core pipeline performance. However, in the full-network scenario, components other than PyPTO also use AICPU, and forcing same-cluster allocation may cause execution timeout due to insufficient AICPU resources.

- Type: string
- Value range: `true` (default, same-cluster allocation), `false` (not forced to the same cluster)

## Configuration Example

```bash
# Disable the same-cluster constraint in the full-network scenario to avoid insufficient AICPU resources.
export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false
```

## Constraints

- When set to `false`, you can use `launch_sched_aicpu_num` (configured through `runtime_options`) to specify the number of available AICPUs.
- When same-cluster allocation is enabled, the `launch_sched_aicpu_num` configuration does not take effect.

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
