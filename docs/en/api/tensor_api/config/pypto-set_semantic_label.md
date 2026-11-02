# pypto.set\_semantic\_label

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:36:22.598Z pushedAt=2026-10-28T11:57:29.614Z nl=2 -->

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

Sets a semantic label for a code segment. PyPTO Toolkit identifies the code lines following the label until the next semantic label is encountered, which helps users locate issues more conveniently.

## Prototype

```python
set_semantic_label(label: str) -> None
```

## Parameters

| Parameter name  | Input/Output | Description                  |
|---------|-----------|-----------------------|
| label   | Input      | Name of the semantic label to set. Supports any string. |

## Return Value

No return value. The setting takes effect immediately upon success.

## Constraints

None

## Examples

```python
pypto.set_semantic_label("kv")
compressed_kv = pypto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
...
```
