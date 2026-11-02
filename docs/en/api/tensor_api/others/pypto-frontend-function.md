# pypto.frontend.function

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T17:51:35.042Z pushedAt=2026-10-28T11:57:30.178Z nl=2 -->

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

`pypto.frontend.function` is used to define reusable computation subgraphs or function modules, allowing the same computation logic to be reused across multiple kernel functions. This feature is intended to improve code modularity and maintainability.

Expected features:

- **Code reuse**: Encapsulate common computation patterns as functions.
- **Modular design**: Combine multiple small functions when building complex operators.
- **Type safety**: Input and output types are explicitly specified in the function signature.
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

No parameters are configured. The decorator is applied directly to the function.

## Return Value

Returns the decorated reusable function object.

## Constraints

1. Only function calls are supported; method calls are not supported.
2. Parameters: Provide the complete number of parameters (no default values) and pass them strictly in the declared order (no out-of-order or keyword arguments).
3. Recursive functions are not supported.

## Differences from pypto.frontend.jit

| Feature | pypto.frontend.jit | pypto.frontend.function |
|------|-------------------|------------------------|
| Purpose | Defines an executable kernel function | Defines a reusable subfunction |
| Compilation | Compiled into a complete computation graph | Inlined or optimized as a subgraph |
| Invocation | Can be called directly from outside | Can only be called within a JIT function |
| Execution | Independent execution unit | Executed by being embedded in the parent function |
