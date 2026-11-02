# pypto.SymbolicScalar Constructor

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:54:39.488Z pushedAt=2026-10-28T11:57:30.192Z nl=2 -->

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

## Description

Creates a new **SymbolicScalar** instance, supporting multiple construction methods.

## Prototype

```python
__init__(self,
         arg0: Union[int, str, 'SymbolicScalar'] = None,
         arg1: Union[int, None] = None
) -> None
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| self    | Input      | Reference to the instance object, automatically passed by Python. |
| arg0    | Input      | Value or name of the symbolic scalar, which can be:<br> - int: Integer value, creating a constant symbolic scalar.<br> - str: Symbol name, creating a symbolic scalar.<br> - SymbolicScalar: Another symbolic scalar, used for copying. |
| arg1    | Input      | Value of the symbolic scalar, optionally used only when arg0 is a string. |

## Return Value

None

## Constraints

- If arg0 is None, a symbolic scalar without an initial value is created.
- If **arg0** is an integer, arg1 is ignored.
- If **arg0** is a string and **arg1** is an integer, a SymbolicScalar with an initial value is created.
- If **arg0** is a string and **arg1** is **None**, a SymbolicScalar without an initial value is created.
- If **arg0** is a SymbolicScalar, its underlying implementation object is copied.

## Examples

```python
a = pypto.SymbolicScalar()
b = pypto.SymbolicScalar(10)
c = pypto.SymbolicScalar("x")
d = pypto.SymbolicScalar("x", 10)
e = pypto.SymbolicScalar(c)
```
