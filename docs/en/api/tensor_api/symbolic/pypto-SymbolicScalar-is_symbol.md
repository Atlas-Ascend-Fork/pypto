# pypto.SymbolicScalar.is\_symbol

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

Determines whether a symbolic scalar is a symbol.

## Prototype

```python
is_symbol(self) -> bool
```

## Parameters

None

## Return Value

Returns True if the symbolic scalar is a symbol. Otherwise, returns False.

## Constraints

A symbolic variable is a variable whose concrete value cannot be determined at runtime and is used to construct a computation graph.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_symbol()
out2 = s2.is_symbol()
```

Result example:

```python
Output data out1: False
Output data out2: True
```
