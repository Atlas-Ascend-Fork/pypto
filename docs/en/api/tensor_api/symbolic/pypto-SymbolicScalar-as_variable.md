# pypto.SymbolicScalar.as\_variable

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:55:22.054Z pushedAt=2026-10-28T11:57:30.201Z nl=2 -->

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

Marks the symbolic scalar as an intermediate variable.

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
- It is typically used to optimize expressions by marking complex expressions as intermediate variables.

## Examples

```python
s = pypto.SymbolicScalar("x")
s.as_variable()  # Mark as an intermediate variable.
```
