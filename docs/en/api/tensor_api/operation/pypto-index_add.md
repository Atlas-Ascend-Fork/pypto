# pypto.index\_add

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T16:51:13.631Z pushedAt=2026-10-28T11:57:29.950Z nl=2 -->

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

This is the non-inplace version of [pypto.index_add_](pypto-index_add_.md).

## Prototype

```python
index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters

See [pypto.index_add_](pypto-index_add_.md) for parameter descriptions.

## Return Value

See [pypto.index_add_](pypto-index_add_.md) for the return value description.

## Constraints

1. See [pypto.index_add_](pypto-index_add_.md) for constraints.

## Examples

See [pypto.index_add_](pypto-index_add_.md) for the call example.
