# pypto.Tensor.view

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T18:24:13.905Z pushedAt=2026-10-28T11:57:30.436Z nl=2 -->

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

## Prototype

```python
view(self, shape: List[int], offsets: List[Union[int, SymbolicScalar]], *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None) -> 'Tensor'
```

## Description

See [pypto.view](../operation/pypto-view.md).
