# pypto_pro.language.printf

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:15:00.949Z pushedAt=2026-10-28T11:57:29.494Z nl=2 -->

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

Prints scalar values or pointers according to a format string during Kernel runtime, for debugging purposes. The format string follows a restricted subset of the C printf syntax.

This API outputs through the device-side printing mechanism, and the specific location is determined by the CANN log configuration of the runtime environment. printf runs on the S (scalar) pipeline and incurs significant runtime overhead. It is intended for debugging only and should be removed in production environments.

## Prototype

```python
pypto_pro.language.printf(
    format_str: str,
    *args,
    loc: bool = False,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| format_str | Input | Format string, which must be a compile-time constant string and cannot be a runtime variable.<br>Supports %d/%i (signed integer), %u (unsigned integer), %x (hexadecimal), %f (floating point), and %p (pointer). Flags, width, precision, and length modifiers are not supported.<br>Unsupported conversions: %% (percent sign itself), %s (string), %c (character). |
| *args | Input | Scalar values or Ptr pointers to print. The number and types must match the format specifiers in the format string. Supports Python int, Python float, Python bool, and Kernel scalar expressions; Ptr pointers are also supported when %p is used.<br>%d/%i: accepts signed integer and DT_BOOL, rejects unsigned integer<br>%u: accepts unsigned integer and DT_BOOL, rejects signed integer<br>%x: accepts unsigned integer, rejects signed integer and DT_BOOL<br>%f: accepts only DT_FP32, rejects other data types<br>%p: accepts only Ptr pointers. |
| loc | Input | Optional. Whether to print the source file/line number before the output. The value is True or False (default). When True, the source file and line number of the call site are printed before the output. |

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl
# Print an integer.
pl.printf("flag=%d, offset=%d\n", flag, offset)

# Print a floating-point value (FP32 only).
pl.printf("value=%f\n", value_f32)

# Print a hexadecimal value.
pl.printf("addr=0x%x\n", addr_u32)

# Plain text (no parameters)
pl.printf("reached checkpoint A\n")

# With source code location
pl.printf("debug: i=%d\n", i, loc=True)
```

Example output:

| Call | Output (assuming flag = 1, offset = 32, value_f32 = 3.14, addr_u32 = 0x1234, i = 5) |
|---|---|
| pypto_pro.language.printf("flag=%d, offset=%d\n", flag, offset) | flag=1, offset=32 |
| pypto_pro.language.printf("value=%f\n", value_f32) | value=3.140000 |
| pypto_pro.language.printf("addr=0x%x\n", addr_u32) | addr=0x1234 |
| pypto_pro.language.printf("reached checkpoint A\n") | reached checkpoint A |
| pypto_pro.language.printf("debug: i=%d\n", i, loc=True) | debug: i=5 (when loc = True, the source code location is additionally attached) |
