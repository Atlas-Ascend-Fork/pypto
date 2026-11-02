# pypto.SymbolicScalar.is\_symbol

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:57:41.656Z pushedAt=2026-10-28T11:57:30.218Z nl=2 -->

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

Determines whether a **SymbolicScalar** is symbolic.

## Prototype

```python
is_symbol(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is symbolic, and **False** otherwise.

## Constraints

A symbolic variable is a variable whose concrete value cannot be determined at runtime and is used to construct a computation graph.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_symbol()
out2 = s2.is_symbol()
```

The results are as follows:

```python
Output data out1: False
Output data out2: True
```
