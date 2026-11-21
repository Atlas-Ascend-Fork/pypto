# pypto.SymbolicScalar.as\_variable

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

Marks a symbolic scalar as an intermediate variable.

## Prototype

```python
as_variable(self) -> None
```

## Parameters

None

## Return Value

None

## Constraints

- This is an in-place operation that modifies the internal state of the symbolic scalar.
- It is usually used to optimize expressions and mark complex expressions as intermediate variables.

## Examples

```python
s = pypto.SymbolicScalar("x")
s.as_variable() # Mark as an intermediate variable.
```
