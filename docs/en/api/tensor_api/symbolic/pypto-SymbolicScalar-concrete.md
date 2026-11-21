# pypto.SymbolicScalar.concrete

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

Obtains the concrete value of a symbolic scalar.

## Prototype

```python
concrete(self) -> int
```

## Parameters

None

## Return Value

Returns the concrete value of the symbolic scalar.

## Constraints

- This method can be called only when is\_concrete\(\) returns True.
- If the symbolic scalar is not concrete, a ValueError exception is thrown.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# If the symbolic scalar does not have a concrete value, an exception is thrown.
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

Result example:

```python
Output data out1: 10
Output data out2: ValueError: Not concrete value is thrown.
```
