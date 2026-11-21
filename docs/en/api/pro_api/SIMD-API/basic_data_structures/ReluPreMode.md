# pypto_pro.language.ReluPreMode

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Enables the elementwise ReLU operation. It is used for data movement from the L0C buffer to the UB, GM, or L1 buffer.

The elementwise ReLU operation sets negative values to zero and leaves positive values unchanged. The computation formula is as follows:

$$dst = \max(src, 0) = \begin{cases} src & src > 0 \\ 0 & src \leq 0 \end{cases}$$

## Prototype

```python
PYPTO_DECLARE_ENUM(ReluPreMode,
    NormalRelu      # Enables the elementwise ReLU operation.
)
```
