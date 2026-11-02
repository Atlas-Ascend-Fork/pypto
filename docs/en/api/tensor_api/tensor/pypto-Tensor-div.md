# pypto.Tensor.div

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:03:16.034Z pushedAt=2026-10-28T11:57:30.269Z nl=2 -->

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
div(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Divisor. <br> Supported types: Tensor, int, float. |
| precision_type  | Input      | Precision type. <br> Supported type: PrecisionType. <br> Default value: PrecisionType.HIGH_PRECISION. <br> HIGH_PRECISION uses higher-precision computation to reduce precision loss; INTRINSIC directly uses chip instructions. |

## Description

See [pypto.div](../operation/pypto-div.md).
