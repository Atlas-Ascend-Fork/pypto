# TopKAlgo

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

TopKAlgo defines the TopK algorithm, which is used to control how TopK computation is processed.

## Prototype

```python
class TopKAlgo(enum.Enum):
     MERGE_SORT = ...    # Merge sort algorithm
     RADIX_SELECT = ...  # Radix selection algorithm
```

## Parameters

| Value| Description|
|:-------|:-----|
| MERGE_SORT | Merge sort algorithm. It sorts the entire tensor and then selects the first k numbers.|
| RADIX_SELECT | Radix selection algorithm. It first finds the kth number and then finds the first k numbers based on the kth number.|

## Suggestions

1. **Default behavior**: If no algorithm is specified, the merge sort algorithm (`MERGE_SORT`) is used by default.
2. **Scenarios with high performance requirements**: The radix selection algorithm (`RADIX_SELECT`) is recommended, with the time complexity of O(n).
3. **Scenarios with low performance requirements**: `MERGE_SORT` can be used, with the time complexity of O(nlogn).

## Examples

```python
import pypto

# Create a tensor.
x = pypto.tensor([2, 3], pypto.DT_FP32)

# Use the merge sort algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.MERGE_SORT)

# Use the radix selection algorithm.
y = pypto.topk(x, 2, -1, True, pypto.TopKAlgo.RADIX_SELECT)

# The merge sort algorithm is used by default.
y = pypto.topk(x, 2, -1, True)
```
