# pypto.SymbolicScalar.is\_immediate

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

Determines whether a symbolic scalar is an immediate value (a concrete numeric value).

## Prototype

```python
is_immediate(self) -> bool
```

## Parameters

None

## Return Value

Returns True if the symbolic scalar is an immediate value. Otherwise, returns False.

## Constraints

An immediate value is a constant whose concrete numeric value can be determined at compile time.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

Result example:

```python
Output data out1: True
Output data out2: False
```
