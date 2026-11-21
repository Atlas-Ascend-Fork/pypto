# pypto.Tensor.remainder

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Prototype

```python
remainder(self, other: 'Tensor | int | float', precision_type: PrecisionType = PrecisionType.HIGH_PRECISION) -> 'Tensor'
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input     | Divisor.<br> Supported types: Tensor, int, and float.|
| precision_type | Input| Precision mode enumeration type, which is used to control the precision mode of the remainder operation. For details, see [PrecisionType](../datatype/PrecisionType.md).<br> The default value is HIGH_PRECISION (high-precision mode).|

## Description

For details, see [pypto.remainder](../operation/pypto-remainder.md).
