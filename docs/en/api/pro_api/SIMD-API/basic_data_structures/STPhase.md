# pypto_pro.language.STPhase

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

Enumerates the phases when the matrix computation results are moved out from the L0C Buffer in phases. It is used by the store, store_tile, move, and insert APIs, and works with the [pypto_pro.language.AccPhase](AccPhase.md) of the matrix computation API to ensure that the result in the L0C Buffer is moved to the GM, UB, or L1 Buffer after computation is complete, and that the storage space can be reused by subsequent computations after the final move-out.

When AccPhase is configured for matmul, matmul_acc, matmul_mx, or matmul_mx_acc, STPhase must also be configured for store, store_tile, move, or insert of the result to be moved out. The last write operation of matrix computation must use AccPhase.Final, and the last move-out of the computation result must use STPhase.Final. When the same computation result is moved out multiple times, STPhase.Partial is used before the last move-out. STPhase of store and store_tile cannot be used together with scale of the tile type.

## Prototype

```python
PYPTO_DECLARE_ENUM(STPhase,
    Unspecified:  # Phased write-back is disabled. This is applicable to the scenario where AccPhase is not configured for the corresponding matrix computation.
    Partial:      # This is not the last write-back of the computation result in the L0C Buffer.
    Final:        # This is the last write-back of the computation result in the L0C Buffer.
)
```
