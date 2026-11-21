# pypto.cond

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

Defines an if condition operation to implement the if function in Python.

## Prototype

```python
cond(scalar: SymInt) -> pypto_impl.RecordIfBranch
```

## Parameters

| Parameter| Input/Output| Description                                                                |
|--------|-----------|----------------------------------------------------------------------|
| scalar | Input     | Condition expression, which can be an integer or symbolic scalar and is used to determine whether the condition is true.|

## Return Value

pypto\_impl.RecordIfBranch: Returns a condition branch object, which is used for the if statement in Python.

## Constraints

- This API must be used together with the if, elif, and else statements in Python.
- The condition expression is recorded in the computational graph.
- Nested conditional statements are supported.
- If the function is not decorated with the @pypto.frontend.jit or @pypto.frontend.function decorator, the condition expression needs to be wrapped with pypto.cond.

## Examples

```python
# No decorator is used. The condition expression needs to be wrapped with pypto.cond.
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_end(s2_idx)):
            ...

# A decorator is used. pypto.cond is not required.
@pypto.frontend.jit
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.is_loop_end(s2_idx):
            ...
```
