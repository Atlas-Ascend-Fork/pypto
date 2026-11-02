# pypto.Tensor.log

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:09:12.779Z pushedAt=2026-10-28T11:57:30.321Z nl=2 -->

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

## Prototype

```python
log(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | Input     | Precision type. <br> Supported type: PrecisionType.<br> The default value is PrecisionType.INTRINSIC. <br> INTRINSIC directly uses chip instructions for computation and is faster; HIGH_PRECISION uses higher-precision computation to reduce precision loss. |

## Description

See [pypto.log](../operation/pypto-log.md).
