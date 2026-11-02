# pypto_pro.language.MemorySpace

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:46:10.933Z pushedAt=2026-10-28T11:57:28.624Z nl=2 -->

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

Memory space enumeration, used to describe the hardware storage area corresponding to a PyPTO data object. When creating an on-chip tile, this enumeration serves as the target_memory attribute of [pypto_pro.language.TileType](TileType.md) to specify the target memory space of the tile.

Different enumeration values correspond to the global storage of the AI processor or the on-chip storage area within the AI Core. For an on-chip tile, the memory space determines the types of computation it can participate in and the supported data movement paths.

## Prototype

```python
class MemorySpace(enum.IntEnum):
    DDR = ...
    Vec = ...
    Mat = ...
    Left = ...
    Right = ...
    Scaling = ...
    Acc = ...
    Bias = ...
    ScaleLeft = ...
    ScaleRight = ...
```

## Parameters

| Parameter Value | Storage Unit | Usage and Constraints |
|---|---|---|
| pypto_pro.language.MemorySpace.DDR | GM | Indicates the storage area where the Kernel input and output tensors reside. This enumeration value is used to indicate the tensor storage location and is not used as the on-chip tile allocation target of make_tile/make_tile_group. |
| pypto_pro.language.MemorySpace.Vec | UB | Working buffer of the Vector Core, used for SIMD computations such as element-wise operations, reduction, sorting, and pure Vector quantization/dequantization. The tile start address must be 32-byte aligned. |
| pypto_pro.language.MemorySpace.Mat | L1 Buffer | Used to store intermediate data of Cube computation. After data is loaded from GM, it can be moved into the L0A Buffer, L0B Buffer, BiasTable Buffer, Fixpipe Buffer, L0A_MX Buffer, and L0B_MX Buffer. The tile start address must be 32-byte aligned. |
| pypto_pro.language.MemorySpace.Left | L0A Buffer | Used to store the left operand of matrix multiplication, which participates in Cube computation together with the right operand in the L0B Buffer. The tile start address must be 512-byte aligned. |
| pypto_pro.language.MemorySpace.Right | L0B Buffer | Used to store the right operand of matrix multiplication, which participates in Cube computation together with the left operand in the L0A Buffer. The tile start address must be 512-byte aligned. |
| pypto_pro.language.MemorySpace.Scaling | Fixpipe Buffer | Used only for per-channel in-line quantization/dequantization parameters of the FIX data path. Data must be moved in through the L1 Buffer: the source tile must have one row, the destination tile data type must be DT_INT64 or DT_UINT64, the destination address and data volume must be 128-byte aligned, and the data volume of a single move must not exceed 4096 bytes. |
| pypto_pro.language.MemorySpace.Acc | L0C Buffer | Used to store the results or intermediate accumulation values of Cube computations such as matmul, matmul_acc, and matmul_mx. The tile start address must be 64-byte aligned. |
| pypto_pro.language.MemorySpace.Bias | BiasTable Buffer | Used to store the bias data of matrix computation. Bias data cannot be loaded directly from GM and must be transferred through the L1 Buffer. When moved in through the L1 Buffer, the source tile must have one row, the destination address and data volume must be 64-byte aligned, and the data volume of a single move must not exceed 4096 bytes. |
| pypto_pro.language.MemorySpace.ScaleLeft | L0A_MX Buffer | Used only to store the left quantization coefficient matrix in MX matrix multiplication. The tile start address must be 32-byte aligned; its address must equal the address of the paired left matrix in the L0A Buffer shifted right by 4 bits. |
| pypto_pro.language.MemorySpace.ScaleRight | L0B_MX Buffer | Used only to store the right quantization coefficient matrix in MX matrix multiplication. The tile start address must be 32-byte aligned; its address must equal the address of the paired right matrix in the L0B Buffer shifted right by 4 bits. |
