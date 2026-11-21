# mask_reg

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

mask_reg is the element-level validity control container for VF computation. It is used to control which elements participate in the operation of a VF operator.

The total bit width of mask_reg is fixed at 256 bits, and its granularity is determined by the associated dtype parameter. When a VF operator is executed, each element's corresponding bit in mask_reg determines whether that element participates in the operation.

- **Bit = 1 (valid)**: The element participates in the operation, and the result is written to the corresponding position in the destination register.
- **Bit = 0 (invalid)**: The element does not participate in the operation, and the corresponding position in the destination register is set to 0. (A few operators, such as vf.full, allow the original value to be retained by setting the mode parameter.)

## Prototype

```python
mask_reg(dtype: DType) -> mask_reg
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dtype | Input| Data type corresponding to the mask granularity, which determines how many bits correspond to one data element. The total bit width of mask_reg is fixed at 256 bits.<br>- vf.mask_reg cannot be directly called. It is automatically declared by the compiler in the assignment form.<br>- mask_reg is created and used within the @pypto_pro.language.vector_function function and is automatically released when the function exits.<br>- The dtype of mask_reg generally matches the dtype of the associated vf.reg_tensor. If they differ, you need to determine the resulting behavior.<br>- The maximum number of MaskReg registers is 16. The compiler automatically reuses registers whose lifetime has ended and the reserved memory. If both registers and the reserved memory have available space, registers are reused first.|

## Constraints

- Data type constraints:

  | dtype | Element Bit Width| Number of Elements| Mask Bits/Element| Total Mask Bits|
  |---|---|---|---|---|
  | DT_INT8 / DT_UINT8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 / DT_HF8 / DT_FP4E2M1 / DT_FP4E1M2 | 8 bits | 256 | 1 bit (b8 granularity)| 256 bits |
  | DT_FP16 / DT_UINT16 / DT_BF16 | 16 bits | 128 | 2 bits (b16 granularity)| 256 bits |
  | DT_FP32 / DT_INT32 / DT_UINT32 | 32 bits | 64 | 4 bits (b32 granularity)| 256 bits |
  | DT_INT64 / DT_UINT64 | 64 bits | 32 | 8 bits (b64 granularity)| 256 bits |

- vf.mask_reg cannot be directly called. It is automatically declared by the compiler in the assignment form, for example, preg = vf.create_mask(...) or preg = vf.eq(...).

## Return Value

Returns a value of type mask_reg.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    reg_out = vf.add(reg, reg, preg)
```
