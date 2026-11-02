# pypto.cond

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:38:44.754Z pushedAt=2026-10-28T11:57:29.627Z nl=2 -->

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

Defines an if conditional operation to implement the if functionality in Python.

## Prototype

```python
cond(scalar: SymInt) -> pypto_impl.RecordIfBranch
```

## Parameters

| Parameter | Input/Output | Description                                                          |
|-----------|--------------|----------------------------------------------------------------------|
| scalar | Input      | Conditional expression, which can be an integer or a SymbolicScalar, used to determine whether the condition is true. |

## Return Value

**pypto\_impl.RecordIfBranch**: Returns a conditional branch object for use in Python if statements.

## Constraints

- Must be used together with Python if, elif, and else statements.
- The conditional expression is recorded into the computation graph.
- Nested conditional statements are supported.
- When the function is not decorated with the **@pypto.frontend.jit** or **@pypto.frontend.function** decorator, the conditional expression must be wrapped with **pypto.cond**.

## Examples

```python
# Without the decorator, wrap the conditional expression with pypto.cond.
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_end(s2_idx, bn_per_batch)):
            ...

# With the decorator, no pypto.cond wrapping is required.
@pypto.frontend.jit
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.is_loop_end(s2_idx, bn_per_batch):
            ...
```
