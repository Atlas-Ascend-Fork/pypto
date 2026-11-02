# pypto.SymbolicScalar.min

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:58:32.267Z pushedAt=2026-10-28T11:57:30.228Z nl=2 -->

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

Computes the minimum of two symbolic scalars.

## Use Cases

Use this method when you need to compare **SymbolicScalar** values, such as dynamic dimension values obtained from `tensor.shape`.

**Difference from pypto.minimum**:

- `pypto.minimum`: Used for element-wise minimum computation on tensors.
- `SymbolicScalar.min`: Used for minimum computation on symbolic scalars.

**Python ternary expressions are not supported**:

```python
# ❌ Error: Ternary expressions are not supported.
cur_seq = kv_act_seqs[b_idx]
tmp = cur_seq - s2_idx * s2_tile
actual = tmp if tmp < s2_tile else s2_tile

# ✅ Correct: Use the .min() method.
actual = (cur_seq - s2_idx * s2_tile).min(s2_tile)
```

## Prototype

```python
min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| other   | Input      | Symbolic scalar or integer to be compared. |

## Return Value

Returns the minimum of the two values.

## Constraints

- If both values are concrete, a concrete constant value is returned.
- If at least one value is not concrete, a symbolic expression is returned.

## Examples

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.min(s2)
out2 = s1.min(3)
s3 = pypto.SymbolicScalar("x")
out3 = s3.min(2)
```

The results are as follows:

```python
Output data out1: 5
Output data out2: 3
Output data out3: RUNTIME_Min(x, 2)
```
