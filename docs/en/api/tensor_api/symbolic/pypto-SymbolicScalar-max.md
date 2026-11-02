# pypto.SymbolicScalar.max

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:57:59.894Z pushedAt=2026-10-28T11:57:30.221Z nl=2 -->

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

Computes the maximum of two symbolic scalars.

## Use Cases

Use this method when you need to compare **SymbolicScalar** values, such as dynamic dimension values obtained from `tensor.shape`.

**Difference from pypto.maximum**:

- `pypto.maximum`: Used for element-wise maximum computation on tensors.
- `SymbolicScalar.max`: Used for maximum computation on symbolic scalars.

**Python ternary expressions are not supported**:

```python
# ❌ Incorrect: The ternary expression is not supported.
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

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Symbolic scalar or integer to be compared. |

## Return Value

Returns the maximum of the two values. If both are concrete values, a constant is returned; otherwise, a symbolic expression of the SymbolicScalar type is returned.

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

The results are as follows:

```python
Output data out1: 10
Output data out2: 13
Output data out3: RUNTIME_Max(x, 2)
```
