# TransMode

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T15:51:54.858Z pushedAt=2026-10-28T11:57:29.702Z nl=2 -->

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

## Description

In scenarios where both the input and output matrices of matmul are of the FP32 data type, this parameter sets whether to enable TF32 computation and the rounding mode for TF32. When enabled, the FP32 data type is converted to the TF32 data type during matrix multiplication. <br> TF32 uses 1 sign bit, 8 exponent bits, and 10 mantissa bits, totaling 19 bits for computation. Fewer mantissa bits reduce the hardware computation workload and accelerate computation, but also introduce precision loss. <br> The input data format remains FP32. The rounding mode is determined by the TransMode parameter, and CAST_ROUND (round to the nearest integer, with ties rounding away from zero) is generally used.

## Prototype

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # Do not enable conversion from the float data type to the TF32 data type.
     CAST_RINT = ...   # Round to the nearest integer, with ties rounding to even.
     CAST_ROUND = ...  # Round to the nearest integer, with ties rounding away from zero.
```
