# DataType

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
<!-- npu="9030" id7 -->
- Kirin 9030 processors: Supported
<!-- end id7 -->
<!-- npu="x90" id8 -->
- Kirin X90 processors: Supported
<!-- end id8 -->
<!-- npu="9050" id11 -->
- Kirin 9050 processors: Supported
<!-- end id11 -->

## Description

DataType is an enumeration class used in the PTO framework to represent the data types of tensors. It defines all supported data types, such as integers, floating-point numbers, and Boolean values. As the core type identifier of tensor operations, DataType is used to specify the storage format and computation precision of tensors.

## Prototype

```python
class DataType(enum.Enum):
     ...  # Enumeration class definition, including all supported data types.

 # Definition of data type constants
 DT_INT4 = ...     # 4-bit signed integer, occupying 0.5 bytes of memory
 DT_INT8 = ...     # 8-bit signed integer, occupying 1 byte of memory
 DT_INT16 = ...    # 16-bit signed integer, occupying 2 bytes of memory
 DT_INT32 = ...    # 32-bit signed integer, occupying 4 bytes of memory
 DT_INT64 = ...    # 64-bit signed integer, occupying 8 bytes of memory
 DT_FP8 = ...      # 8-bit floating-point number, used for low-precision computation
 DT_FP16 = ...     # 16-bit half-precision floating-point number, occupying 2 bytes of memory
 DT_FP32 = ...     # 32-bit single-precision floating-point number, occupying 4 bytes of memory
 DT_BF16 = ...     # 16-bit brain floating-point number, occupying 2 bytes of memory
 DT_HF4 = ...      # 4-bit half-precision floating-point number, occupying 0.5 bytes of memory
 DT_HF8 = ...      # 8-bit half-precision floating-point number, occupying 1 byte of memory
 DT_FP4_E2M1 = ...  # 4-bit floating-point number, 2-bit exponent and 1-bit mantissa, and two elements occupying 1 byte of memory
 DT_FP4_E1M2 = ...  # 4-bit floating-point number, 1-bit exponent and 2-bit mantissa, and two elements occupying 1 byte of memory
 DT_FP8E4M3 = ...  # 8-bit floating-point number, 4-bit exponent and 3-bit mantissa, and occupying 1 byte of memory
 DT_FP8E5M2 = ...  # 8-bit floating-point number, 5-bit exponent and 2-bit mantissa, and occupying 1 byte of memory
 DT_FP8E8M0 = ...  # 8-bit floating-point number, 8-bit exponent and 0-bit mantissa, and occupying 1 byte of memory
 DT_FP4_E2M1X2 = ... # MXFP4 format, 2-bit exponent and 1-bit mantissa, two elements packed, and occupying 1 byte of memory
 DT_FP4_E1M2X2 = ... # MXFP4 format, 1-bit exponent and 2-bit mantissa, two elements packed, and occupying 1 byte of memory
 DT_UINT8 = ...    # 8-bit unsigned integer, occupying 1 byte of memory
 DT_UINT16 = ...   # 16-bit unsigned integer, occupying 2 bytes of memory
 DT_UINT32 = ...   # 32-bit unsigned integer, occupying 4 bytes of memory
 DT_UINT64 = ...   # 64-bit unsigned integer, occupying 8 bytes of memory
 DT_BOOL = ...     # Boolean type, occupying 1 byte of memory
 DT_DOUBLE = ...   # 64-bit double-precision floating-point number, occupying 8 bytes of memory
```

## Constraints

The supported data types vary depending on the product models.
<!-- npu="950" id6 -->
- Ascend 950PR&950DT products: All data types are supported.
<!-- end id6 -->
<!-- npu="A3" id4 -->
- Atlas A3 products: DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 are not supported.
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 products: DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 are not supported.
<!-- end id5 -->
<!-- npu="9030" id9 -->
- Kirin 9030 processors: DT_INT4, DT_INT64, DT_FP8, DT_BF16, DT_HF4, DT_HF8, DT_FP4_E2M1, DT_FP8E4M3, DT_FP8E5M2, DT_FP8E8M0, DT_FP4_E2M1X2, DT_FP4_E1M2X2, DT_UINT16, DT_UINT32, DT_UINT64, and DT_DOUBLE are not supported.
<!-- end id9 -->
<!-- npu="x90" id10 -->
- Kirin X90 processors: DT_INT4, DT_INT64, DT_FP8, DT_BF16, DT_HF4, DT_HF8, DT_FP4_E2M1, DT_FP8E4M3, DT_FP8E5M2, DT_FP8E8M0, DT_FP4_E2M1X2, DT_FP4_E1M2X2, DT_UINT16, DT_UINT32, DT_UINT64, and DT_DOUBLE are not supported.
<!-- end id10 -->
<!-- npu="9050" id12 -->
- Kirin 9050 processors: DT_INT4, DT_INT64, DT_FP8, DT_BF16, DT_HF4, DT_HF8, DT_FP4_E2M1, DT_FP8E4M3, DT_FP8E5M2, DT_FP8E8M0, DT_FP4_E2M1X2, DT_FP4_E1M2X2, DT_UINT16, DT_UINT32, DT_UINT64, and DT_DOUBLE are not supported.
<!-- end id12 -->
