# pypto_pro.language.MemorySpace

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

Enumerates the memory space. It is used to describe the hardware storage area corresponding to a PyPTO data object. When an on-chip tile is being created, this enumeration is used as the target_memory attribute of [pypto_pro.language.TileType](TileType.md) to specify the target memory space of the tile.

Different enumerated values correspond to the global storage of the AI processor or the on-chip storage area within the AI Core. For on-chip tiles, the memory space determines the types of computations that can be performed and the supported data movement paths.

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

| Value| Storage Unit| Usage and Constraints|
|---|---|---|
| pypto_pro.language.MemorySpace.DDR | GM | Indicates the storage area where the input and output tensors of a kernel are located. This enumerated value is used to indicate the tensor storage location and is not used as the on-chip tile allocation target of make_tile/make_tile_group.|
| pypto_pro.language.MemorySpace.Vec | UB | Serves as the working buffer of the Vector Core. It is used for SIMD computations such as elementwise operations, reduction, sorting, and pure Vector quantization/dequantization. The start address of the tile must be 32-byte aligned.|
| pypto_pro.language.MemorySpace.Mat | L1 Buffer | Stores intermediate data of Cube computations. Data can be moved from the GM to the L0A Buffer, L0B Buffer, BiasTable Buffer, Fixpipe Buffer, L0A_MX Buffer, and L0B_MX Buffer. The start address of the tile must be 32-byte aligned.|
| pypto_pro.language.MemorySpace.Left | L0A Buffer | Stores the left operand of matrix multiplication and is used together with the right operand in the L0B Buffer for Cube computations. The start address of the tile must be 512-byte aligned.|
| pypto_pro.language.MemorySpace.Right | L0B Buffer | Stores the right operand of matrix multiplication and is used together with the left operand in the L0A Buffer for Cube computations. The start address of the tile must be 512-byte aligned.|
| pypto_pro.language.MemorySpace.Scaling | Fixpipe Buffer | Stores only per-channel quantization/dequantization parameters in the FIX data path. This space is not the scale storage area of the quant/dequant API. Data needs to be moved in from the L1 Buffer: The source tile must have one row, the destination tile must be of the DT_INT64 or DT_UINT64 type, the destination address and data volume must be 128-byte aligned, and the volume of data moved at a time cannot exceed 4096 bytes.|
| pypto_pro.language.MemorySpace.Acc | L0C Buffer | Stores the results or intermediate accumulated values of Cube computations such as matmul, matmul_acc, and matmul_mx. The start address of the tile must be 64-byte aligned.|
| pypto_pro.language.MemorySpace.Bias | BiasTable Buffer | Stores the bias data for matrix computations. The bias data cannot be directly loaded from the GM and must be transferred through the L1 Buffer. When the data is moved in from the L1 Buffer, the source tile must have one row, the destination address and data volume must be 64-byte aligned, and the volume of data moved at a time cannot exceed 4096 bytes.|
| pypto_pro.language.MemorySpace.ScaleLeft | L0A_MX Buffer | Stores only the left quantization coefficient matrix in MX matrix multiplication. The start address of the tile must be 32-byte aligned. The address must be equal to the address of the left matrix in the L0A Buffer right-shifted by four bits.|
| pypto_pro.language.MemorySpace.ScaleRight | L0B_MX Buffer | Stores only the right quantization coefficient matrix in MX matrix multiplication. The start address of the tile must be 32-byte aligned. The address must be equal to the address of the right matrix in the L0B Buffer right-shifted by four bits.|
