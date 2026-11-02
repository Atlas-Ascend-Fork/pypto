# pypto_pro.language.DataType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:44:23.404Z pushedAt=2026-10-28T11:57:28.606Z nl=2 -->

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

Enumeration of data types.

## Prototype

```python
DT_FP4 = DataType.FP4               # 4-bit floating-point number, 2-bit exponent, 1-bit mantissa, two bits occupy 1 byte of memory
DT_FP8E4M3FN = DataType.FP8E4M3FN   # 8-bit floating-point number, 4-bit exponent, 3-bit mantissa, occupying 1 byte of memory
DT_FP8E5M2 = DataType.FP8E5M2       # 8-bit floating-point number, 5-bit exponent, 2-bit mantissa, occupying 1 byte of memory
DT_FP8E8M0 = DataType.FP8E8M0       # 8-bit floating-point number, 8-bit exponent, 0-bit mantissa, occupying 1 byte of memory
DT_FP4E2M1 = DataType.FP4E2M1       # 4-bit floating-point number, 2-bit exponent, 1-bit mantissa, two bits occupy 1 byte of memory
DT_FP4E1M2 = DataType.FP4E1M2       # 4-bit floating-point number with 1 exponent bit and 2 mantissa bits. Two bits occupy 1 byte of memory.
DT_FP16 = DataType.FP16             # 16-bit half-precision floating-point number, occupying 2 bytes of memory.
DT_FP32 = DataType.FP32             # 32-bit single-precision floating-point number, occupying 4 bytes of memory.
DT_BF16 = DataType.BF16             # 16-bit Brain Float format, occupying 2 bytes of memory.
DT_HF4 = DataType.HF4               # 4-bit HiFloat format. Two bits occupy 1 byte of memory.
DT_HF8 = DataType.HF8               # 8-bit HiFloat format, occupying 1 byte of memory.
DT_INT4 = DataType.INT4             # 4-bit signed integer, occupying 1 byte of memory.
DT_INT8 = DataType.INT8             # 8-bit signed integer, occupying 1 byte of memory.
DT_INT16 = DataType.INT16           # 16-bit signed integer, occupying 2 bytes of memory.
DT_INT32 = DataType.INT32           # 32-bit signed integer, occupying 4 bytes of memory.
DT_INT64 = DataType.INT64           # 64-bit signed integer, occupying 8 bytes of memory.
DT_UINT4 = DataType.UINT4           # 4-bit unsigned integer, occupying 1 byte of memory.
DT_UINT8 = DataType.UINT8           # 8-bit unsigned integer, occupying 1 byte of memory.
DT_UINT16 = DataType.UINT16         # 16-bit unsigned integer, occupying 2 bytes of memory.
DT_UINT32 = DataType.UINT32         # 32-bit unsigned integer, occupying 4 bytes of memory.
DT_UINT64 = DataType.UINT64         # 64-bit unsigned integer, occupying 8 bytes of memory.
DT_BOOL = DataType.BOOL             # Boolean type, occupying 1 byte of memory.
```

The DT_XXX constants above are Python-side aliases. DataType itself is not a Python enum but a C++ class exposed to Python under the same name through binding. Each DataType.XXX is a static constant instance of that class.

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

- A bare integer constant (such as 42) uses DT_INT64. DT_UINT64 is used only when the value exceeds the upper limit of DT_INT64. A bare floating-point constant (such as 3.14) uses DT_FP32.

- Data elements of MXFP8 use DT_FP8E4M3FN or DT_FP8E5M2, and data elements of MXFP4 use DT_FP4E2M1 or DT_FP4E1M2. Both use DT_FP8E8M0 to store the group scaling factor. For details, see [matmul_mx](../cube_computation/matmul_mx.md) and [matmul_mx_acc](../cube_computation/matmul_mx_acc.md).

## Common APIs

| Method | Description |
|---|---|
| dtype.get_bit() / dtype.bits() | Gets the bit width (for example, pypto_pro.language.DT_FP16.get_bit() returns 16). |
| dtype.is_float() | Whether the type is a floating-point type. |
| dtype.is_int() | Whether the type is an integer type (signed or unsigned). |
| dtype.is_signed_int() / dtype.is_signed() | Whether the type is a signed integer type. |
| dtype.is_unsigned_int() / dtype.is_unsigned() | Whether the type is an unsigned integer type. |
| dtype.to_string() / str(dtype) | Human-readable name (for example, "fp16"). |
| dtype.to_c_type_string() | C type string (for example, "half"). |
