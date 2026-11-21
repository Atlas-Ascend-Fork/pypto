# pypto_pro.language.pto_assert

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

Runtime assertion: When the condition is false, an error message is printed to the device log. Plaintext and printf-style formatted messages are supported. If the condition is true, the assertion passes silently, and there is no output.

> [!CAUTION]
> In the current NPU implementation, if the assertion fails, only a log is printed, and the kernel execution is not terminated nor an exception is thrown on the host. The actual behavior depends on the execution result of the target device.

## Prototype

```python
pypto_pro.language.pto_assert(
    condition: bool,
    /,
    format_str: Optional[str] = None,
    *args,
    loc: bool = False,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| condition | Input| Assertion condition, which must be a scalar Boolean value (with dtype of BOOL). The Python True/False value is also accepted. If a non-Boolean scalar (such as INT32) is used, TypeError will be reported. The source code text of the condition expression is automatically extracted by the compiler and is used for the default output.|
| format_str | Input| (Optional) Error message format string. It must be a compile-time constant and cannot be a runtime variable. If this parameter is not provided, an empty message is used. In this case, the additional *args is not written into the IR and is not involved in the output. The output is fixed to Assertion failed: <source code of the condition expression>. If this parameter is provided, the assertion failure information is output first, followed by a line of formatted message. The format specifier rules are the same as those of [pypto_pro.language.printf](printf.md).|
| *args | Input| (Optional) Parameter value in the format string. Python int, Python float, Python bool, and kernel scalar expressions are supported. When %p is used, Ptr (pointer) is also supported. The number and type of the arguments must match the format specifiers in format_str.|
| loc | Input| (Optional) Whether to print the source file/line number before the output. The value can be True or False (default).|

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl
# Only condition
pl.pto_assert(flag)

# Condition + Plain text message
pl.pto_assert(flag, "flag is false")

# Condition + Formatted message
pl.pto_assert(offset != 2, "offset=%d", offset)

# With source code location
pl.pto_assert(flag, "unexpected state", loc=True)
```

Example output (when the condition is false):

| Call| Output|
|---|---|
| pypto_pro.language.pto_assert(flag) | Assertion failed: flag |
| pypto_pro.language.pto_assert(flag, "flag is false") | Assertion failed: flag<br>flag is false |
| pypto_pro.language.pto_assert(offset != 2, "offset=%d", offset) | Assertion failed: offset != 2<br>offset=2 (assuming that the offset value is 2)|
| pypto_pro.language.pto_assert(flag, "unexpected state", loc=True) | Assertion failed: flag<br>unexpected state |

> If the condition is true, there is no output, and the assertion passes silently.
