# OpType

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

OpType defines the type of a comparison operation, which is used to specify the operation type for comparing two symbolic scalars. It is mainly used in condition judgment and logical operation.

## Prototype

```python
class OpType(enum.Enum):
     EQ =...  # Equal to operation, which is used to determine whether two values are equal.
     NE = ...  # Not equal to operation, which is used to determine whether two values are not equal.
     LT =...  # Less than operation, which is used to determine whether the left value is less than the right value.
     LE =...  # Less than or equal to operation, which is used to determine whether the left value is less than or equal to the right value.
     GT =...   # Greater than operation, which is used to determine whether the left value is greater than the right value.
     GE =...  # Greater than or equal to operation, which is used to determine whether the left value is greater than or equal to the right value.
```
