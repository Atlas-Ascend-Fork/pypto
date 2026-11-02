# TopKAlgo

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:51:00.212Z pushedAt=2026-10-28T11:57:29.692Z nl=2 -->

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

**TopKAlgo** defines the TopK algorithm and controls how TopK computation is processed.

## Prototype

```python
class TopKAlgo(enum.Enum):
     MERGE_SORT = ...    # Merge sort algorithm.
     RADIX_SELECT = ...  # Radix select algorithm.
```

## Parameters

| parameter value | Description |
|:-------|:-----|
| MERGE_SORT | Merge sort algorithm. Sorts the entire tensor and then selects the top k values. |
| RADIX_SELECT | Radix select algorithm. First finds the k-th value, and then selects the top k values based on it. |

## Usage Recommendations

1. **Default behavior**: If no algorithm is specified, the `MERGE_SORT` mode is used by default.
2. **Performance-sensitive scenarios**: The `RADIX_SELECT` mode is recommended, with a time complexity of O(n).
3. **Scenarios with low performance requirements**: The `MERGE_SORT` mode can be used, with a time complexity of O(n log n).

## Example

```python
import pypto

# Create a tensor.
x = pypto.tensor([2, 3], pypto.DT_FP32)

# Use the merge sort algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)

# Use the radix select algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.RADIX_SELECT)

# Use the merge sort algorithm by default.
y = pypto.topk(x, 2, -1, True)
```
