# DataType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:44:37.626Z pushedAt=2026-10-28T11:57:29.660Z nl=2 -->

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

## Description

**DataType** is an enumeration class in the PTO framework used to represent tensor data types. It defines all supported data types, including integers, floating-point numbers, and Boolean values. As the core type identifier for tensor operations, **DataType** specifies the storage format and computation precision of a tensor.

## Prototype

```python
class DataType(enum.Enum):
     ...  # Enum class definition, containing all supported data types.

 # Data type constant definitions.
 DT_INT4 = ...     # 4-bit signed integer, occupying 0.5 bytes of memory.
 DT_INT8 = ...     # 8-bit signed integer, occupying 1 byte of memory.
 DT_INT16 = ...    # 16-bit signed integer, occupying 2 bytes of memory.
 DT_INT32 = ...    # 32-bit signed integer, occupying 4 bytes of memory.
 DT_INT64 = ...    # 64-bit signed integer, occupying 8 bytes of memory.
 DT_FP8 = ...      # 8-bit floating-point number, used for low-precision computation.
 DT_FP16 = ...     # 16-bit half-precision floating-point number, occupying 2 bytes of memory.
 DT_FP32 = ...     # 32-bit single-precision floating-point number, occupying 4 bytes of memory.
 DT_BF16 = ...     # 16-bit Brain Float format, occupying 2 bytes of memory.
 DT_HF4 = ...      # 4-bit Half Float format, occupying 0.5 bytes of memory.
 DT_HF8 = ...      # 8-bit Half Float format, occupying 1 byte of memory.
 DT_FP4E2M1 = ...  # 4-bit floating-point number with 2 exponent bits and 1 mantissa bit; two elements occupy 1 byte of memory.
 DT_FP8E4M3 = ...  # 8-bit floating-point number with 4 exponent bits and 3 mantissa bits, occupying 1 byte of memory.
 DT_FP8E5M2 = ...  # 8-bit floating-point number with 5 exponent bits and 2 mantissa bits, occupying 1 byte of memory.
 DT_FP8E8M0 = ...  # 8-bit floating-point number with 8 exponent bits and 0 mantissa bits, occupying 1 byte of memory
 DT_FP4_E2M1X2 = ... # MXFP4 format with 2 exponent bits and 1 mantissa bit, packed as two elements, occupying 1 byte of memory
 DT_FP4_E1M2X2 = ... # MXFP4 format with 1 exponent bit and 2 mantissa bits, packed as two elements, occupying 1 byte of memory
 DT_UINT8 = ...    # 8-bit unsigned integer, occupying 1 byte of memory
 DT_UINT16 = ...   # 16-bit unsigned integer, occupying 2 bytes of memory
 DT_UINT32 = ...   # 32-bit unsigned integer, occupying 4 bytes of memory
 DT_UINT64 = ...   # 64-bit unsigned integer, occupying 8 bytes of memory
 DT_BOOL = ...     # Boolean type, occupying 1 byte of memory
 DT_DOUBLE = ...   # 64-bit double-precision floating-point number, occupying 8 bytes of memory
```

## Constraints

The supported data types vary by model:
<!-- npu="950" id6 -->
- Ascend 950PR&950DT products: All data types are supported.
<!-- end id6 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 are not supported.
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 products: DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 are not supported.
<!-- end id5 -->
