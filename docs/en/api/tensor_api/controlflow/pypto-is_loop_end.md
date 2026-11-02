# pypto.is\_loop\_end

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:41:46.113Z pushedAt=2026-10-28T11:57:29.640Z nl=2 -->

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

Determines whether the current iteration is the end of the loop.

## Prototype

```python
is_loop_end(scalar: SymInt) -> SymbolicScalar
```

## Parameters

| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| scalar  | Input      | Index of the current loop. |

## Return Value

Returns a symbolic scalar expression indicating whether it is the end of the loop (a Boolean value).

## Constraints

- **scalar** must be a symbolic scalar returned by the loop iterator.
- If it is not a loop index, a **ValueError** exception is thrown.
- When the function is not decorated with the **@pypto.frontend.jit** or **@pypto.frontend.function** decorator, the conditional expression must be wrapped with **pypto.cond**.

## Examples

```python
# No decorator is used, so the conditional expression must be wrapped with pypto.cond.
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.cond(pypto.is_loop_end(idx)):
            ...

# A decorator is used, so wrapping with pypto.cond is not required.
@pypto.frontend.jit
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.is_loop_end(idx):
            ...
```
