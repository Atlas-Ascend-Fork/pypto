# pypto_pro.language.printf

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

Prints scalar values or pointers based on the format string for debugging when the kernel is running. The format string complies with the restricted subset of the C printf syntax.

This API outputs data through the printing mechanism on the device. The specific location to view the output is determined by the CANN log configuration in the operating environment. printf runs on the S (scalar) pipeline and has significant runtime overhead. It is used only for debugging and should be removed from the production environment.

## Prototype

```python
pypto_pro.language.printf(
    format_str: str,
    *args,
    loc: bool = False,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| format_str | Input| Format string, which must be a constant string at compile time and cannot be a variable at runtime.<br>The following types are supported: %d/%i (signed integer), %u (unsigned integer), %x (hexadecimal number), %f (floating-point number), and %p (pointer). Modifiers such as flags, width, precision, and length are not supported.<br>The following conversions are not supported: %% (percentage sign itself), %s (string), and %c (character).|
| *args | Input| Scalar value or pointer to be printed. The number and type of the arguments must match the format specifiers in the format string. Python int, Python float, Python bool, and kernel scalar expressions are supported. When %p is used, Ptr (pointer) is also supported.<br>%d/%i: Accepts signed integers and DT_BOOL, and rejects unsigned integers.<br>%u: Accepts unsigned integers and DT_BOOL, and rejects signed integers.<br>%x: Accepts unsigned integers, and rejects signed integers and DT_BOOL.<br>%f: Accepts only DT_FP32 and rejects other data types.<br>%p: Accepts only pointers.|
| loc | Input| (Optional) Whether to print the source file/line number before the output. The value can be True or False (default). If the value is True, the source file and line number of the call location are printed before the output.|

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl
# Print integers.
pl.printf("flag=%d, offset=%d\n", flag, offset)

# Prints floating-point numbers (FP32 only).
pl.printf("value=%f\n", value_f32)

# Print in hexadecimal.
pl.printf("addr=0x%x\n", addr_u32)

# Plain text (without parameters)
pl.printf("reached checkpoint A\n")

# With source code location
pl.printf("debug: i=%d\n", i, loc=True)
```

Example output:

| Call| Output (assuming flag = 1, offset = 32, value_f32 = 3.14, addr_u32 = 0x1234, i = 5)|
|---|---|
| pypto_pro.language.printf("flag=%d, offset=%d\n", flag, offset) | flag=1, offset=32 |
| pypto_pro.language.printf("value=%f\n", value_f32) | value=3.140000 |
| pypto_pro.language.printf("addr=0x%x\n", addr_u32) | addr=0x1234 |
| pypto_pro.language.printf("reached checkpoint A\n") | reached checkpoint A |
| pypto_pro.language.printf("debug: i=%d\n", i, loc=True) | debug: i=5 (The source code location is attached when loc is set to True.)|
