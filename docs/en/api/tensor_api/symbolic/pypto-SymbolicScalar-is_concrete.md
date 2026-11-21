# pypto.SymbolicScalar.is\_concrete

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

Determines whether a symbolic scalar has a concrete value.

## Prototype

```python
is_concrete(self) -> bool
```

## Parameters

None

## Return Value

Returns True if there is a concrete value. Otherwise, returns False.

## Constraints

- The constant value is always concrete.
- Some expressions may also have concrete values under certain conditions.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.is_concrete()
s2 = pypto.SymbolicScalar("x")
out2 = s2.is_concrete()
```

Result example:

```python
Output data out1: True
Output data out2: False
```
