# pypto.SymbolicScalar Constructor

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
<!-- npu="9030" id4 -->
- Kirin 9030 processors: Supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Supported
<!-- end id6 -->

## Description

Creates a new SymbolicScalar instance. Multiple construction modes are supported.

## Prototype

```python
__init__(self,
         arg0: Union[int, str, 'SymbolicScalar'] = None,
         arg1: Union[int, None] = None
) -> None
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| self    | Input     | Instance object reference, which is automatically passed by Python.|
| arg0    | Input     | Value or name of the symbolic scalar. The options are as follows:<br> - int: Integer value, used to create a constant symbolic scalar.<br> - str: Symbolic name, used to create a symbolic scalar.<br> - SymbolicScalar: Another symbolic scalar, used for replication.|
| arg1    | Input     | Value of the symbolic scalar, which can be used only when arg0 is a string.|

## Return Value

None

## Constraints

- If arg0 is None, a symbolic scalar without an initial value is created.
- If arg0 is an integer, arg1 will be ignored.
- If arg0 is a string and arg1 is an integer, a symbolic scalar with an initial value is created.
- If arg0 is a string and arg1 is None, a symbolic scalar without an initial value is created.
- If arg0 is SymbolicScalar, the underlying implementation object is copied.

## Examples

```python
a = pypto.SymbolicScalar()
b = pypto.SymbolicScalar(10)
c = pypto.SymbolicScalar("x")
d = pypto.SymbolicScalar("x", 10)
e = pypto.SymbolicScalar(c)
```
