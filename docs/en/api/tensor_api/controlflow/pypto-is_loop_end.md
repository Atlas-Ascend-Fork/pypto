# pypto.is\_loop\_end

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

Determines whether the current iteration is the end of a loop.

## Prototype

```python
is_loop_end(scalar: SymInt) -> SymbolicScalar
```

## Parameters

| Parameter | Input/Output| Description                                                                |
|---------|-----------|----------------------------------------------------------------------|
| scalar  | Input     | Current loop index.|

## Return Value

Returns a symbolic scalar expression indicating whether the current iteration is the end of the loop (a Boolean value).

## Constraints

- scalar must be a symbolic scalar returned by the loop iterator.
- If the index is not a loop index, a ValueError exception is thrown.
- If the function is not decorated with the @pypto.frontend.jit or @pypto.frontend.function decorator, the condition expression needs to be wrapped with pypto.cond.

## Examples

```python
# No decorator is used. The condition expression needs to be wrapped with pypto.cond.
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.cond(pypto.is_loop_end(idx)):
            ...

# A decorator is used. pypto.cond is not required.
@pypto.frontend.jit
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.is_loop_end(idx):
            ...
```
