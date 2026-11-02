# pypto_pro.language.pto_assert

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T15:15:56.039Z pushedAt=2026-10-28T11:57:29.505Z nl=2 -->

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

Runtime assertion: prints an error message to the device log when the condition is false. Supports plain text and printf-style formatted messages. Passes silently with no output when the condition is true.

> [!CAUTION] Note
> In the current NPU implementation, a failed assertion only prints a log and does not abort Kernel execution or throw a Host-side exception. The actual behavior is subject to the execution result on the target device.

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

| Parameter | Input/Output | Description |
|---|---|---|
| condition | Input | Assertion condition, which must be a scalar Boolean value (dtype BOOL). Python True/False is also accepted. A non-Boolean scalar (such as INT32) raises a TypeError. The source text of the condition expression is automatically extracted by the compiler for the default output. |
| format_str | Input | Optional. Error message format string, which must be a compile-time constant and cannot be a runtime variable. If not provided, an empty message is used, in which case the extra *args are not written to the IR and do not participate in the output; the output is fixed as `Assertion failed: <conditional expression source code>`. If provided, the assertion failure information is output first, followed by an additional line of formatted message. The format specifier rules are the same as those of [pypto_pro.language.printf](printf.md). |
| *args | Input | Optional. Parameter values in the format string. Python int, Python float, Python bool, and Kernel scalar expressions are supported; Ptr pointers are also supported when %p is used. The number and types must match the format specifiers in format_str. |
| loc | Input | Optional. Whether to print the source file/line number before the output. The value is True or False (default). |

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl
# Condition only.
pl.pto_assert(flag)

# Condition + plain text message.
pl.pto_assert(flag, "flag is false")

# Condition + formatted message.
pl.pto_assert(offset != 2, "offset=%d", offset)

# With source code location.
pl.pto_assert(flag, "unexpected state", loc=True)
```

Example output (when the condition is false):

| Call | Output |
|---|---|
| pypto_pro.language.pto_assert(flag) | Assertion failed: flag |
| pypto_pro.language.pto_assert(flag, "flag is false") | Assertion failed: flag<br>flag is false |
| pypto_pro.language.pto_assert(offset != 2, "offset=%d", offset) | Assertion failed: offset != 2<br>offset=2 (assuming the value of offset is 2) |
| pypto_pro.language.pto_assert(flag, "unexpected state", loc=True) | Assertion failed: flag<br>unexpected state |

> When the condition is true, there is no output and the assertion passes silently.
