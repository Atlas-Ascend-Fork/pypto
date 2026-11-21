# OutType

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

OutType defines the output data type, which is used to specify the output format of some operations (such as comparison operations) and distinguish Boolean value output from bit value output.

## Prototype

```python
class OutType(enum.Enum):
     BOOL =...  # Boolean output type. The output is True or False.
     BIT =...   # Bit output type. The bit value of 0 or 1 is output.
```
