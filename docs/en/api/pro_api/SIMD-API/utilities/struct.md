# pypto_pro.language.struct

<!-- md-trans-meta sourceCommit=d15a7740ea679c63c4ade0c9eaf473865b3b634e translatedAt=2026-10-26T14:55:36.530Z pushedAt=2026-10-28T11:57:29.337Z nl=2 -->

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

Creates a named struct variable whose field layout is determined at compile time. The field declaration order is consistent with the keyword argument order, and can be used to organize a small amount of metadata such as batch numbers, block numbers, and address offsets.

## Prototype

```python
pypto_pro.language.struct(
    *args: Any,
    **kwargs: Any,
) -> Any
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| \*args | Input | Struct type name. Only one string constant can and must be passed. The name can contain only letters, digits, and underscores, cannot start with a digit, and cannot be a C++ keyword. |
| \*\*kwargs | Input | Struct member variables, passed in the form field=value. At least one member variable must be specified. Expanding parameters through a dictionary is not supported, for example, struct("run_info", \*\*fields).<br>- Member variable names must not be duplicated.<br>- Member variable names can contain only letters, digits, and underscores, cannot start with a digit, and cannot be a C++ keyword.<br>- Member variables support only the following types:<br>&nbsp;&nbsp;- **Scalar**: The initial value supports integers, floating-point numbers, booleans, or Scalar expressions.<br>&nbsp;&nbsp;- **One-dimensional scalar array**: A one-dimensional non-empty scalar array. The array length and element type must be determined at compile time, and the elements should be of the same data type. Nested named structs (created through make_tuple or struct) are not supported as member variables.<br>- The value of a scalar member variable can be modified through arr.field = value, and array member variables can be read and written through arr.field[index]. Only assignments of the same type are supported. |

## Constraints

- When assigning a struct, the struct type name, and the names, order, scalar types, and array lengths of the member variables on both sides of the equation must be exactly the same. For example, if the same variable is assigned a struct in the if and else branches respectively, the two structs must meet the above conditions.
- In control flow (if/else/for/while), if a struct is created by value copy, the result is an independent copy, and modifying it does not affect the original struct.
- When accessing elements of a one-dimensional array field by subscript, the subscript must be within the valid index range of that field.

## Return Value

Returns a named struct variable.

## Examples

### Reading and Writing Array Fields in a Loop

```python
import pypto_pro.language as pl


@pl.jit()
def struct_field_kernel(out: pl.Tensor[[5], pl.DT_INT32]):
    # Create a struct with an array field.
    s = pl.struct("RunInfo", batch_id=0, offsets=[0, 0, 0, 0])

    with pl.section_vector():
        # Assign a value to an array field element (s.arr_field[idx] = val).
        s.offsets[0] = 10
        s.offsets[1] = 20
        s.offsets[2] = 30
        s.offsets[3] = 40

        # Read an array field element (s.arr_field[idx]).
        total = 0
        for i in pl.range(0, 4):
            total = total + s.offsets[i]
        pl.setval(out, 0, s.offsets[0])
        pl.setval(out, 1, s.offsets[3])
        pl.setval(out, 2, total)
        pl.setval(out, 3, s.batch_id)
        pl.setval(out, 4, s.offsets[1] + s.offsets[2])
```
