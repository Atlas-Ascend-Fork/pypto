# pypto_pro.language.get_subblock_num

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:36:25.878Z pushedAt=2026-10-28T11:57:29.209Z nl=2 -->

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

Obtains the total number of subblocks in the current block (that is, the number of slave cores associated with a block, also known as the task ration).

## Prototype

```python
pypto_pro.language.get_subblock_num() -> int
```

## Parameters

None

## Constraints

None

## Return Value

Returns the total number of subblocks of the DT_INT64 type. The return value depends on the core type and compilation mode:

- **AIC core**: Always returns 1 (AIC is a block and has no AIC slave cores).
- **AIV core**:
  - Fused operator (mix, AIC:AIV = 1:2): Returns 2 (each AI Core contains two AIV slave cores).
  - Pure Vector operator (aiv-only): Returns 1 (AIV is a block and has no subblock division).

## Examples

In a fused operator, get_block_idx() returns a logical index on the AIV core (block_idx * subblock_num + subblock_idx). Dividing it by get_subblock_num() recovers the physical AI Core index, so that the Cube and Vector sides can use a unified core_id to partition data:

```python
import pypto_pro.language as pl

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    # The AIC/AIV sides obtain the same physical core index. See the NOTE below.
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    with pl.section_cube():
        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            ...  # The Cube side performs load/matmul/store by row block i.
    with pl.section_vector():
        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            ...  # The Vector side partitions data with the same core_id, aligned with the Cube side.


matmul_example[None, NUM_CORES](a, b, out)
```

> [!NOTE]
> On the AIC core, this division is block_idx // 1; on the AIV core, it is (block_idx * 2 + subblock_idx) // 2. Both yield the same AI Core index, so the Cube and Vector sides can share the same core_id for data partitioning.
