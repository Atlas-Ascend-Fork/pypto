# pypto.Tensor.fmod

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:05:31.181Z pushedAt=2026-10-28T11:57:30.292Z nl=2 -->

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
fmod(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Divisor. <br> Supported types: Tensor, int, float. |
| precision_type | Input | Precision mode enumeration, used to control the precision mode of the modulo operation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br> Defaults to HIGH_PRECISION (high-precision mode). |

## Description

See [pypto.fmod](../operation/pypto-fmod.md).
