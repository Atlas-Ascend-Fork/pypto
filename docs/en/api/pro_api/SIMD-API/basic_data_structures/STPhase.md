# pypto_pro.language.STPhase

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:48:00.867Z pushedAt=2026-10-28T11:57:28.637Z nl=2 -->

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

Stage enumeration used when the matrix computation result is written back to GM in stages. It is used by the store and store_tile APIs and works with [pypto_pro.language.AccPhase](AccPhase.md) of the matrix computation API to ensure that the result in the L0C Buffer is written back after the computation completes, and that the storage space can be reused by subsequent computations after the final write-back.

When the corresponding matmul, matmul_acc, matmul_mx, or matmul_mx_acc configures AccPhase, store or store_tile must also configure STPhase. The last write operation of the matrix computation must use AccPhase.Final, and the last write-back of the computation result must use STPhase.Final. When performing multiple write-backs of the same computation result, use STPhase.Partial for all write-backs before the last one. STPhase cannot be used together with scale of the Tile type.

## Prototype

```python
PYPTO_DECLARE_ENUM(STPhase,
    Unspecified,  # Staged write-back is not enabled. This applies to scenarios where the corresponding matrix computation does not configure AccPhase.
    Partial,      # This is not the last write-back of the computation result in this L0C Buffer.
    Final         # This is the last write-back of the computation result in this L0C Buffer.
)
```
