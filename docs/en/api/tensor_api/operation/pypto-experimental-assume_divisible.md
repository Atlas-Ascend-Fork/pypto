# pypto.experimental.assume_divisible

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T16:31:51.793Z pushedAt=2026-10-28T11:57:29.873Z nl=2 -->

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

Declares that a symbolic scalar expression is guaranteed to be divisible by a specified integer at runtime. This API returns a SymbolicScalar representing expr: a SymbolicScalar input is returned as is, and an int input is converted to a SymbolicScalar. At the same time, the API registers the normalized compilation optimization assumption on the Program. For example, when a dynamic length vm is known to be an integer multiple of the tile size, the compiler can use this to simplify the per-tile dynamic valid_shape, thereby enabling optimizations such as dualdst that depend on a static valid shape.

## Prototype

```python
assume_divisible(expr: SymbolicScalar | int, divisor: int) -> SymbolicScalar
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| expr | Input | Symbolic scalar expression or integer constant for which the divisibility relationship is declared. |
| divisor | Input | Divisibility factor, which must be a positive integer. |

## Return Value

Returns a SymbolicScalar representing expr. A SymbolicScalar input is returned as is, and an int input is converted to a SymbolicScalar.

## Constraints

- By default, this is a semantic contract without runtime checking. You must guarantee that expr % divisor == 0 always holds at runtime; violating this may lead to silently incorrect results.
- When runtime_debug_mode=4 is enabled, the compiler converts each assumption expression explicitly registered and normalized by you into a modulo assertion at the control flow entry. The generated code evaluates this expression using the actual runtime values and reports an error on violation. Compiler-derived expressions that are not explicitly registered through assume_divisible are not checked.
- The API only provides optimization information. It does not change the value of expr, nor does it replace the dynamic expression with divisor.
- Once registered on a Program, it can be queried by all Functions in the same Program.
- A ValueError is raised when a compile-time constant is not divisible; when it is divisible, this is a trivial no-op.
- A ValueError is raised when divisor <= 0.

## Examples

```python
tile_m = 128
valid_m = pypto.experimental.assume_divisible(a.shape[0], tile_m)
tile = pypto.view(mm, [tile_m, 128], [0, 0], valid_shape=[valid_m, 128])
```
