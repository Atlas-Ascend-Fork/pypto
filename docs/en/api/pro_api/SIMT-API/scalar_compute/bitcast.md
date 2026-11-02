# pypto_pro.language.simt.bitcast

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:03:58.151Z pushedAt=2026-10-28T11:57:29.403Z nl=2 -->

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

Reinterprets the binary bit pattern of the source operand as the specified equal-width data type, without performing numeric conversion, rounding, or saturation.

## Prototype

```python
pypto_pro.language.simt.bitcast(
    value: Scalar,
    dtype: DType,
) -> Scalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| value | Input | Source operand of the Scalar type. Tensor or Tile elements must be accessed by subscript before being passed in. |
| dtype | Input | Destination data type of the DType type. For supported data type conversions, see the following table. |

The destination data types supported for each source data type are as follows:

| Source Data Type | Destination Data Type |
|---|---|
| DT_FP16 | DT_INT16, DT_UINT16 |
| DT_BF16 | DT_INT16, DT_UINT16 |
| DT_FP32 | DT_INT32, DT_UINT32 |
| DT_INT16 | DT_FP16, DT_BF16 |
| DT_UINT16 | DT_FP16, DT_BF16 |
| DT_INT32 | DT_FP32 |
| DT_UINT32 | DT_FP32 |

## Constraints

- This API can only be called in a SIMT entry function or helper function defined by @pypto_pro.language.vector_function(mode="simt").

- The source data type and the destination data type must have the same bit width. Conversion between identical data types is not supported.

- NaN, infinity, negative zero, and invalid floating-point encodings are all preserved with their original bit patterns without normalization.

## Return Value

Returns a Scalar that has the same binary bit pattern as value and the data type dtype.

## Examples

```python
import pypto_pro.language as pl


@pl.vector_function(mode="simt", max_threads=256)
def fp32_bits(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    output[0, tid] = pl.simt.bitcast(source[0, tid], pl.DT_UINT32)


@pl.jit()
def simt_fp32_bits_kernel(
    source: pl.Tensor[[1, 256], pl.DT_FP32],
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        fp32_bits[256](source, output)
```
