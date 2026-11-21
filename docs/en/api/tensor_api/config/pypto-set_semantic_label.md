# pypto.set\_semantic\_label

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

Sets the semantic label of a code segment. PyPTO Toolkit will identify the code lines following this label until it encounters the next semantic label. This helps locate problems more easily.

## Prototype

```python
set_semantic_label(label: str) -> None
```

## Parameters

| Parameter | Input/Output| Description                 |
|---------|-----------|-----------------------|
| label   | Input     | Sets the name of the semantic label. Any string is supported.|

## Return Value

No value is returned. The setting takes effect immediately after the operation is successful.

## Constraints

None

## Examples

```python
pypto.set_semantic_label("kv")
compressed_kv = pypto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
...
```
