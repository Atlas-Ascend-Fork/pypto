# PYPTO_LAUNCH_SCHED_SAME_CLUSTER

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:12:26.238Z pushedAt=2026-10-27T02:33:00.919Z nl=2 -->

## Description

Controls whether to forcibly allocate scheduling threads (AICPU) within the same cluster. Allocating scheduling threads within the same cluster can achieve better inter-core pipeline performance. However, in the full-network scenario, components other than PyPTO also use AICPU, and forcing same-cluster allocation may cause execution timeout due to insufficient AICPU resources.

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
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id3 -->
