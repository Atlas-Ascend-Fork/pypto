# pypto.Tensor.exp

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:03:42.695Z pushedAt=2026-10-28T11:57:30.279Z nl=2 -->

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
exp(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | Input      | Precision type. <br> Supported types: PrecisionType. <br> Default value: PrecisionType.INTRINSIC. <br> INTRINSIC directly uses chip instructions for computation, which is faster; HIGH_PRECISION uses higher-precision computation to reduce precision loss. |

## Description

See [pypto.exp](../operation/pypto-exp.md).
