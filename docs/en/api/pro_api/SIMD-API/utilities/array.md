# pypto_pro.language.array

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

Declares a fixed-length, mutable array of scalars whose element dtype and length are both
compile-time constants. The array is kernel-local storage. Reading `A[i]` yields a scalar of the
element dtype, and assigning `A[i] = value` writes one element. Use it to keep a small,
compile-time-sized set of scalars within a kernel, such as counters or temporary indices, that must
persist across statements or loop iterations. If such scalars must travel with a struct, use the
one-dimensional array fields of [pypto_pro.language.struct](./struct.md).

## Prototype

```python
pypto_pro.language.array(size: int, /, *, dtype: Any) -> Any
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| size | Input| Positive compile-time element count. The value must be an integer literal or a kernel-local constant.|
| dtype | Input| Scalar element dtype, for example pypto_pro.language.DT_INT32. Supported dtypes are pypto_pro.language.DT_INT8, pypto_pro.language.DT_INT16, pypto_pro.language.DT_INT32, pypto_pro.language.DT_INT64, pypto_pro.language.DT_UINT8, pypto_pro.language.DT_UINT16, pypto_pro.language.DT_UINT32, pypto_pro.language.DT_UINT64, pypto_pro.language.DT_FP16, pypto_pro.language.DT_FP32, pypto_pro.language.DT_BF16, and pypto_pro.language.DT_BOOL. Low-precision storage-only types, such as pypto_pro.language.DT_INT4 and pypto_pro.language.DT_FP8E4M3FN, are not supported.|

## Constraints

- The storage is uninitialized. A kernel must write each element before its first read.
- `A[0]` and `A[i]` accept one integer index. An out-of-range constant index is rejected at compile
  time. For a runtime index, the kernel must keep the value in `[0, size)`.
- The index must be an integer scalar. Multi-dimensional subscripts are not supported.
- A load returns a scalar of the element dtype. `A[i] = value` accepts a scalar value; a literal is
  converted to the element dtype, and any other scalar is cast by codegen, following the
  pypto_pro.language.setval contract.
- A = pypto_pro.language.array(...) must bind the declaration to a variable. Assigning that variable
  to another name aliases the same storage; writing through either name modifies the same array.
- Each pypto_pro.language.array(...) call creates independent storage, even when the arguments are
  identical.
- An array declared outside a loop retains its element values across iterations. An array declared
  inside a loop body has that body's lifetime.
- The storage is local to the core body that uses it. Do not use one array across different core
  sections, such as writing it in pypto_pro.language.section_cube() and reading it in
  pypto_pro.language.section_vector().
- Arrays cannot be used inside pypto_pro.language.vector_function.

## Return Value

Returns the declared array value bound to a kernel-local variable.

## Examples

### Alias and Scalar Capture

```python
import pypto_pro.language as pl


@pl.jit()
def alias_kernel(out: pl.Tensor[[3], pl.DT_INT32]):
    values = pl.array(8, dtype=pl.DT_INT32)
    values[0] = 7
    a = values[0]          # captures the int32 value 7
    alias = values         # refers to the same storage
    alias[0] = 9
    values[1] = a          # values[0] is 9, values[1] is 7
    with pl.section_vector():
        pl.setval(out, 0, a)
        pl.setval(out, 1, values[0])
        pl.setval(out, 2, values[1])
```

### Fill in One Loop and Read Back in Another

```python
import pypto_pro.language as pl


@pl.jit()
def loop_kernel(out: pl.Tensor[[8], pl.DT_INT32]):
    values = pl.array(8, dtype=pl.DT_INT32)
    for i in pl.range(0, 8):
        values[i] = i * 2 + 1
    with pl.section_vector():
        for i in pl.range(0, 8):
            pl.setval(out, i, values[i])
```
