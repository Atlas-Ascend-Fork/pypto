# pypto.SymbolicScalar.concrete

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:56:17.584Z pushedAt=2026-10-28T11:57:30.205Z nl=2 -->

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

- This method can be called only when **is_concrete()** returns **True**.
- If the symbolic scalar is not concrete, a **ValueError** exception is thrown.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# An exception is thrown if the symbolic scalar has no concrete value.
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

The results are as follows:

```python
Output data out1: 10
Output data out2: Throws exception ValueError: Not concrete value
```
