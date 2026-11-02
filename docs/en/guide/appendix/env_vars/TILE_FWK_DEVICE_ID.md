# TILE_FWK_DEVICE_ID

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-27T01:13:21.798Z pushedAt=2026-10-27T02:34:26.005Z nl=2 -->

## Description

Specifies the NPU device ID used for PyPTO operator execution. The framework and test cases determine the target device through this variable.

- Type: integer
- Value range: `0` to `N-1` (N is the number of available NPUs)

## Configuration Example

```bash
export TILE_FWK_DEVICE_ID=0
```

## Constraints

- The specified device ID must be available. You can run `npu-smi info` to verify this.
- In multi-process scenarios, avoid using the same device ID across multiple processes.

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
