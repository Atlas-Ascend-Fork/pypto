# pypto_pro.language.simt.bitcast

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

Reinterprets the binary bit pattern of the source operand as the specified fixed-width data type without performing numeric conversion, rounding, or saturation.

## Prototype

```python
pypto_pro.language.simt.bitcast(
    value: Union[int, float],
    dtype: DType,
) -> Scalar
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| value | Input| Source operand, which can be an integer or floating-point constant, or a runtime scalar expression of the corresponding type. The tensor or tile element needs to be accessed by subscript before being passed.|
| dtype | Input| Destination data type, which is of the DType type. The following table lists the supported data type conversions.|

The destination data types supported by each source data type are listed below.

| Source Data Type| Destination Data Type|
|---|---|
| DT_FP16 | DT_INT16, DT_UINT16|
| DT_BF16 | DT_INT16, DT_UINT16|
| DT_FP32 | DT_INT32, DT_UINT32|
| DT_INT16 | DT_FP16, DT_BF16|
| DT_UINT16 | DT_FP16, DT_BF16|
| DT_INT32 | DT_FP32 |
| DT_UINT32 | DT_FP32 |

## Constraints

- This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

- The bit widths of the source and destination data types must be the same. Conversion between the same data types is not supported.

- NaN, infinities, negative zeros, and invalid floating-point encodings are all preserved in their original bit patterns and are not normalized.

## Return Value

Returns a scalar that has the same binary bit pattern as value and a data type specified by dtype.

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
