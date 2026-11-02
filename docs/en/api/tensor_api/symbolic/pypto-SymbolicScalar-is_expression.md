# pypto.SymbolicScalar.is\_expression

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:56:41.920Z pushedAt=2026-10-28T11:57:30.211Z nl=2 -->

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

Determines whether a symbolic scalar is an expression.

## Prototype

```python
is_expression(self) -> bool
```

## Parameters

None

## Return Value

Returns **True** if it is an expression, and **False** otherwise.

## Constraints

An expression is formed by combining multiple symbols or constants through operations.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
s3 = s2 + 5
out1 = s1.is_expression()
out2 = s2.is_expression()
out3 = s3.is_expression()
```

The results are as follows:

```python
Output data out1: False
Output data out2: False
Output data out3: True
```
