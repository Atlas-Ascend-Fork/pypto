# OpType

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T15:47:44.167Z pushedAt=2026-10-28T11:57:29.670Z nl=2 -->

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

**OpType** defines the types of comparison operations, used to specify the operation type when performing comparison operations between symbolic scalars. It is mainly used in conditional judgment and logical operations.

## Prototype

```python
class OpType(enum.Enum):
     EQ = ...  # Equal operation. Determines whether two values are equal.
     NE = ...  # Not equal operation. Determines whether two values are not equal.
     LT = ...  # Less than operation. Determines whether the left value is less than the right value.
     LE = ...  # Less than or equal operation. Determines whether the left value is less than or equal to the right value.
     GT = ...  # Greater than operation. Determines whether the left value is greater than the right value.
     GE = ...  # Greater than or equal operation. Determines whether the left value is greater than or equal to the right value.
```
