# pypto.index\_add

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Serves as the non-inplace version of [pypto.index_add_](pypto-index_add_.md).

## Prototype

```python
index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters

See [pypto.index_add_](pypto-index_add_.md) for parameter descriptions.

## Return Value

See [pypto.index_add_](pypto-index_add_.md) for return value descriptions.

## Constraints

1. See [pypto.index_add_](pypto-index_add_.md) for constraints.

## Examples

See [pypto.index_add_](pypto-index_add_.md) for call examples.
