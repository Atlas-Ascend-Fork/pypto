# pypto.SymbolicScalar.is\_immediate

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:57:18.467Z pushedAt=2026-10-28T11:57:30.214Z nl=2 -->

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

Determines whether a symbolic scalar is an immediate value (a concrete numeric value).

## Prototype

```python
is_immediate(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is an immediate value, and **False** otherwise.

## Constraints

An immediate value is a constant whose concrete numeric value can be determined at compile time.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

The results are as follows:

```python
Output data out1: True
Output data out2: False
```
