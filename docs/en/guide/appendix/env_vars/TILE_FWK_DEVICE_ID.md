# TILE_FWK_DEVICE_ID

<!-- md-trans-meta sourceCommit=50972549b8df8d831de411545a92c8acac5eac27 translatedAt=2026-11-15T01:30:52.015Z pushedAt=2026-11-18T06:03:01.288Z nl=2 -->

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
