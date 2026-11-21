# pypto.frontend.function

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

`pypto.frontend.function` is used to define reusable subgraphs or function modules, allowing the same computation logic to be reused in multiple kernel functions. This feature aims to improve the modularity and maintainability of code.

Expected features:

- **Code reuse**: Encapsulates common computation patterns as functions.
- **Modular design**: Combines multiple small functions when complex operators are constructed.
- **Type safety**: Specifies the input and output types in the function signature.
- **Automatic optimization**: The compiler can inline or optimize function calls.

## Prototype

```python
@pypto.frontend.function
def reusable_function(
    arg1: pypto.Tensor,
    arg2: pypto.Tensor,
    ...
) -> pypto.Tensor:
    ...
```

## Parameters

No parameters are configured. The decorator is directly applied to the function.

## Return Value

Returns the decorated reusable function object.

## Constraints

1. Only function name calls are supported. Method calls are not supported.
2. Parameters: Complete parameters (without default values) are passed in strict accordance with the declaration sequence (without out-of-order or keyword arguments).
3. Recursive functions are not supported.

## Differences from pypto.frontend.jit

| Feature| pypto.frontend.jit | pypto.frontend.function |
|------|-------------------|------------------------|
| Purpose| Defines executable kernel functions.| Defines reusable subfunctions.|
| Compilation| Compiled into a complete computational graph.| Inlined or optimized as a subgraph.|
| Call method| Can be directly called from outside.| Can be called only within the JIT function.|
| Execution| Independent execution unit| Embedded in the parent function for execution.|
