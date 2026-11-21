# pypto.Tensor.sqrt

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
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Prototype

```python
sqrt(self, precision_type: PrecisionType = PrecisionType.INTRINSIC) -> 'Tensor'
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| precision_type  | Input     | Precision type.<br> Supported type: PrecisionType.<br> The default value is PrecisionType.INTRINSIC.<br> INTRINSIC uses the chip instructions for computation, which is faster. HIGH_PRECISION uses higher-precision computation to reduce precision loss.|

## Description

For details, see [pypto.sqrt](../operation/pypto-sqrt.md).
