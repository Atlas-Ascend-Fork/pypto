# pypto.SymbolicScalar.is\_concrete

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:56:25.944Z pushedAt=2026-10-28T11:57:30.208Z nl=2 -->

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

Determines whether a symbolic scalar has a concrete value.

## Prototype

```python
is_concrete(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it has a concrete value; otherwise, returns **False**.

## Constraints

- Constant values are always concrete.
- Certain expressions may also have concrete values under specific conditions.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.is_concrete()
s2 = pypto.SymbolicScalar("x")
out2 = s2.is_concrete()
```

The results are as follows:

```python
Output data out1: True
Output data out2: False
```
