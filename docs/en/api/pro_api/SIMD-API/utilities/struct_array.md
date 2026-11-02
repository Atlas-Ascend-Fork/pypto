# pypto_pro.language.struct_array

<!-- md-trans-meta sourceCommit=d15a7740ea679c63c4ade0c9eaf473865b3b634e translatedAt=2026-10-26T14:53:53.934Z pushedAt=2026-10-28T11:57:29.324Z nl=2 -->

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

A struct array, which stores N identical structs accessed by index.

## Prototype

```python
pypto_pro.language.struct_array(
    size: int,
    *args: Any,
    **kwargs: Any,
) -> Any
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| size | Input | Array length.<br>- Must be a compile-time constant positive integer. |
| \*args | Input | Struct type name. Only one string constant can be passed. The name can contain only letters, digits, and underscores, cannot start with a digit, and cannot be a C++ keyword. |
| \*\*kwargs | Input | Struct member variables, passed in the form field=value, with at least one member variable specified. Expanding parameters through a dictionary is not supported, for example, struct_array(2, "run_info", \*\*fields).<br>- Member variable names must not be duplicated.<br>- A member variable name can contain only letters, digits, and underscores, cannot start with a digit, and cannot be a C++ keyword.<br>- Member variables support only the following types:<br>&nbsp;&nbsp;- **Scalar**: The initial value supports an integer, a floating-point number, a Boolean value, or a Scalar expression.<br>&nbsp;&nbsp;- **One-dimensional scalar array**: A non-empty one-dimensional scalar array whose length and element type must be determined at compile time, and whose elements should be of the same data type. Nested named structs (created through make_tuple or struct) are not supported as member variables.<br>- The value of a scalar member variable can be modified through arr[i].field = value, and an array member variable can be read from or written to through arr[i].field[index]. Only assignment of the same type is supported. |

## Constraints

- When assigning a struct, the struct type names on both sides of the equation, as well as the names, order, scalar types, and array lengths of the member variables, must be exactly the same. For example, if the same variable is assigned a struct in the if and else branches respectively, the two structs must meet the preceding conditions.
- In control flow (if/else/for/while), if a struct is created by value copy, an independent copy is obtained, and modifying it does not affect the original struct.
- When accessing an element of a struct array or a one-dimensional array field through a subscript, the subscript must be within the valid index range of the corresponding array.

## Return Value

Returns a struct array object. For each element, the scalar field can be accessed through arr[i].field, and the array field element can be accessed through arr[i].field[j].

## Examples

### Read and Write Fields by Index

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # Create a 2-slot struct array: scalar field + array field.
    run_infos = pl.struct_array(2, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    # Modify the scalar field by index.
    run_infos[0].batch_id = 7
    run_infos[1].batch_id = 9

    # Modify the array field element by index.
    run_infos[0].innerS1Realsize[3] = 128
    run_infos[1].innerS1Realsize[1] = 64

    # Read back the array field element and output it.
    with pl.section_vector():
        pl.setval(out, 0, run_infos[0].batch_id)
        pl.setval(out, 1, run_infos[1].batch_id)
        pl.setval(out, 2, run_infos[0].innerS1Realsize[3])
        pl.setval(out, 3, run_infos[1].innerS1Realsize[1])
```

### Reading and Writing Array Fields in a Loop

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_field_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # Create a 4-slot struct array with an array field.
    run_infos = pl.struct_array(4, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    with pl.section_vector():
        # Assign a value to an array field element (arr[i].field[j] = val).
        for i in pl.range(0, 4):
            run_infos[i].innerS1Realsize[0] = i * 100

        # Read an array field element (arr[i].field[j]).
        for i in pl.range(0, 4):
            pl.setval(out, i, run_infos[i].innerS1Realsize[0])
```
