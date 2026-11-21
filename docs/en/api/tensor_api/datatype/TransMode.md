# TransMode

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
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

## Description

Sets whether to enable TF32 computation and the rounding mode of TF32 when the input and output matrices of matmul are of the FP32 data type. Once enabled, the FP32 data type is converted to the TF32 data type during matrix multiplication.<br> TF32 uses one sign bit, eight exponent bits, and 10 mantissa bits, totaling 19 bits for computation. Fewer mantissa bits reduce the hardware computation workload and accelerate computation, but also cause precision loss.<br> The input data format is still FP32. The rounding mode is determined by the TransMode parameter. Generally, CAST_ROUND (rounding to the nearest integer, and rounding away from zero in case of a tie) is used.

## Prototype

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # Do not convert the float data type to the TF32 data type.
     CAST_RINT = ...   # Round to the nearest integer. In case of a tie, round to the nearest even integer.
     CAST_ROUND = ...  # Round to the nearest integer. In case of a tie, round away from zero.
```
