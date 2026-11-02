# pypto_pro.language.ReluPreMode

<!-- md-trans-meta sourceCommit=6dc68739e530164e836f473e2668f6a3df7ad7fd translatedAt=2026-10-26T12:46:51.466Z pushedAt=2026-10-28T11:57:28.630Z nl=2 -->

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

Enumeration for enabling the in-line ReLU operation, used in the L0C buffer to UB data transfer scenario.

The in-line ReLU sets negative values to zero element-wise while leaving positive values unchanged. The formula is as follows:

$$dst = \max(src, 0) = \begin{cases} src & src > 0 \\ 0 & src \leq 0 \end{cases}$$

## Prototype

```python
PYPTO_DECLARE_ENUM(ReluPreMode,
    NormalRelu      # Enable the in-line ReLU operation.
)
```
