# pypto_pro.language.DataType

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

Enumerates data types.

## Prototype

```python
DT_FP4 = DataType.FP4               # 4-bit floating-point number, 2-bit exponent and 1-bit mantissa, and two elements occupying 1 byte of memory
DT_FP8E4M3FN = DataType.FP8E4M3FN   # 8-bit floating-point number, 4-bit exponent and 3-bit mantissa, and occupying 1 byte of memory
DT_FP8E5M2 = DataType.FP8E5M2       # 8-bit floating-point number, 5-bit exponent and 2-bit mantissa, and occupying 1 byte of memory
DT_FP8E8M0 = DataType.FP8E8M0       # 8-bit floating-point number, 8-bit exponent and 0-bit mantissa, and occupying 1 byte of memory
DT_FP4E2M1 = DataType.FP4E2M1       # 4-bit floating-point number, 2-bit exponent and 1-bit mantissa, and two elements occupying 1 byte of memory
DT_FP4E1M2 = DataType.FP4E1M2       # 4-bit floating-point number, 1-bit exponent and 2-bit mantissa, and two elements occupying 1 byte of memory
DT_FP16 = DataType.FP16             # 16-bit half-precision floating-point number, occupying 2 bytes of memory
DT_FP32 = DataType.FP32             # 32-bit single-precision floating-point number, occupying 4 bytes of memory
DT_BF16 = DataType.BF16             # 16-bit brain floating-point number, occupying 2 bytes of memory
DT_HF4 = DataType.HF4               # 4-bit HiFloat number, two elements occupying 1 byte of memory
DT_HF8 = DataType.HF8               # 8-bit HiFloat number, occupying 1 byte of memory
DT_INT4 = DataType.INT4             # 4-bit signed integer, occupying 1 byte of memory
DT_INT8 = DataType.INT8             # 8-bit signed integer, occupying 1 byte of memory
DT_INT16 = DataType.INT16           # 16-bit signed integer, occupying 2 bytes of memory
DT_INT32 = DataType.INT32           # 32-bit signed integer, occupying 4 bytes of memory
DT_INT64 = DataType.INT64           # 64-bit signed integer, occupying 8 bytes of memory
DT_UINT4 = DataType.UINT4            # 4-bit unsigned integer, occupying 1 byte of memory
DT_UINT8 = DataType.UINT8           # 8-bit unsigned integer, occupying 1 byte of memory
DT_UINT16 = DataType.UINT16         # 16-bit unsigned integer, occupying 2 bytes of memory
DT_UINT32 = DataType.UINT32         # 32-bit unsigned integer, occupying 4 bytes of memory
DT_UINT64 = DataType.UINT64         # 64-bit unsigned integer, occupying 8 bytes of memory
DT_BOOL = DataType.BOOL             # Boolean type, occupying 1 byte of memory
```

The preceding DT_XXX constants are aliases on the Python side. DataType is not a Python enumeration, but a C++ class that is bound and exposed to Python as a type with the same name. Each DataType.XXX is a static constant instance of the class.

```C++
class DataType {
public:
    static const DataType BOOL;      // Boolean (true/false)
    static const DataType INT4;      // 4-bit signed integer
    static const DataType INT8;      // 8-bit signed integer
    static const DataType INT16;     // 16-bit signed integer
    static const DataType INT32;     // 32-bit signed integer
    static const DataType INT64;     // 64-bit signed integer
    static const DataType UINT4;     // 4-bit unsigned integer
    static const DataType UINT8;     // 8-bit unsigned integer
    static const DataType UINT16;    // 16-bit unsigned integer
    static const DataType UINT32;    // 32-bit unsigned integer
    static const DataType UINT64;    // 64-bit unsigned integer
    static const DataType FP4;       // 4-bit floating point
    static const DataType FP8E4M3FN; // 8-bit floating point (IEEE 754 e4m3fn format)
    static const DataType FP8E5M2;   // 8-bit floating point (IEEE 754 e5m2 format)
    static const DataType FP8;       // 8-bit floating point (backward compatibility alias)
    static const DataType FP8E8M0;   // 8-bit floating point (8-bit exponent, 0-bit mantissa)
    static const DataType FP4E2M1;   // 4-bit floating point (2-bit exponent, 1-bit mantissa)
    static const DataType FP4E1M2;   // 4-bit floating point (1-bit exponent, 2-bit mantissa)
    static const DataType FP16;      // 16-bit floating point (IEEE 754 half precision)
    static const DataType FP32;      // 32-bit floating point (IEEE 754 single precision)
    static const DataType FP64;      // 64-bit floating point (IEEE 754 double precision)
    static const DataType BF16;      // 16-bit brain floating point
    static const DataType HF4;       // 4-bit Hisilicon float
    static const DataType HF8;       // 8-bit Hisilicon float
    static const DataType INDEX;     // 64-bit index type
}
```

## Constraints

- Use DT_INT64 for bare integer constants (such as 42). Use DT_UINT64 only when the value exceeds the upper limit of DT_INT64. Use DT_FP32 for raw floating-point constants (such as 3.14).

- The data elements of MXFP8 are stored in DT_FP8E4M3FN or DT_FP8E5M2, and the data elements of MXFP4 are stored in DT_FP4E2M1 or DT_FP4E1M2. Both of them use DT_FP8E8M0 to store the group scaling factor. For details, see [matmul_mx](../cube_computation/matmul_mx.md) and [matmul_mx_acc](../cube_computation/matmul_mx_acc.md).

## Common APIs

| Method| Description|
|---|---|
| dtype.get_bit() / dtype.bits() | Obtains the bit width (for example, pypto_pro.language.DT_FP16.get_bit() returns 16).|
| dtype.is_float() | Whether it is a floating-point type.|
| dtype.is_int() | Whether it is an integer type (signed or unsigned).|
| dtype.is_signed_int() / dtype.is_signed() | Whether it is a signed integer.|
| dtype.is_unsigned_int() / dtype.is_unsigned() | Whether it is an unsigned integer.|
| dtype.to_string() / str(dtype) | Human-readable name (for example, "fp16").|
| dtype.to_c_type_string() | C-type string (such as "half").|
