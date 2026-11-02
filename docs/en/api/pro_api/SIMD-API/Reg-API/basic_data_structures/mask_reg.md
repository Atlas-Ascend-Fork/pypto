# mask_reg

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T13:27:29.039Z pushedAt=2026-10-28T11:57:28.870Z nl=2 -->

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

The mask register (mask_reg) is a container that controls element-level validity in VF computation. It controls which elements participate in the operation of a VF operator.

The total bit width of mask_reg is fixed at 256 bits, and its granularity is determined by the associated dtype parameter. When a VF operator executes, whether each data element participates in the operation is determined by the corresponding bit in mask_reg:

- **Bit is 1 (valid)**: The element participates in the operation, and the result is written to the corresponding position in the destination register.
- **Bit 0 (invalid)**: The element does not participate in the operation, and the corresponding position in the destination register is set to zero (a few operators such as vf.full support retaining the original value through the mode parameter).

## Prototype

```python
mask_reg(dtype: DType) -> mask_reg
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dtype | Input | Data type corresponding to the mask granularity, which determines how many bits correspond to one data element. The total bit width of mask_reg is fixed at 256 bits.<br>- vf.mask_reg cannot be called directly; it is automatically declared by the compiler in assignment form.<br>- mask_reg is created and used within the @pypto_pro.language.vector_function function and is automatically released after the function ends.<br>- The dtype of mask_reg is generally the same as that of the associated vf.reg_tensor. If they differ, you need to determine the resulting behavior yourself.<br>- The maximum number of MaskReg registers is 16. The compiler automatically reuses registers whose lifetime has ended and reserved memory. If both registers and reserved memory have available space, registers are reused first. |

## Constraints

- Data type constraints:

  | dtype | Element Bit Width | Number of Elements | Mask Bits per Element | Total Mask Bits |
  |---|---|---|---|---|
  | DT_INT8 / DT_UINT8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 / DT_HF8 / DT_FP4E2M1 / DT_FP4E1M2 | 8 bit | 256 | 1 bit (b8 granularity) | 256 bit |
  | DT_FP16 / DT_UINT16 / DT_BF16 | 16 bit | 128 | 2 bit (b16 granularity) | 256 bit |
  | DT_FP32 / DT_INT32 / DT_UINT32 | 32 bit | 64 | 4 bit (b32 granularity) | 256 bit |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 | 8 bit (b64 granularity) | 256 bit |

- vf.mask_reg cannot be called directly. It is automatically declared by the compiler in an assignment form, such as preg = vf.create_mask(...) or preg = vf.eq(...).

## Return Value

Returns the mask_reg type.

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    reg_out = vf.add(reg, reg, preg)
```
