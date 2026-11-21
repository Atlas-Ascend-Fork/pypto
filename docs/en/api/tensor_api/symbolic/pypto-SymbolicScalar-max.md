# pypto.SymbolicScalar.max

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

Computes the larger value between two symbolic scalars.

## Application Scenario

This method is used when you need to compare symbolic scalars (such as dynamic dimension values obtained from `tensor.shape`).

**It differs from pypto.maximum in the following ways:**

- `pypto.maximum`: Computes the maximum elementwise values of tensors.
- `SymbolicScalar.max`: Computes the maximum values of symbolic scalars.

**Python ternary expressions are not supported:**

```python
# ❌ Incorrect: Ternary expressions are not supported.
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp > threshold else threshold

# ✅ Correct: Use the .max() method.
actual = (cur_seq - s2_idx * s2_tile).max(threshold)
```

## Prototype

```python
max(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input     | Symbolic scalar or integer to be compared.|

## Return Value

Returns the larger value between two values. If both values are concrete, a constant is returned. Otherwise, a symbolic expression of the SymbolicScalar type is returned.

## Constraints

- If both values are concrete, a concrete constant value is returned.
- If at least one value is not concrete, a symbolic expression is returned.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.max(s2)
out2 = s1.max(13)
s3 = pypto.SymbolicScalar("x")
out3 = s3.max(2)
```

Result example:

```python
Output data out1: 10
Output data out2: 13
Output data out3: RUNTIME_Max(x, 2)
```
