# pypto.set_conv_tile_shapes

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
- Kirin 9030 processors: Not supported
<!-- end id4 -->
<!-- npu="x90" id5 -->
- Kirin X90 processors: Not supported
<!-- end id5 -->
<!-- npu="9050" id6 -->
- Kirin 9050 processors: Not supported
<!-- end id6 -->

## Description

Sets the TileShape size of each dimension in the L1/L0 buffer hierarchy during convolution (conv) computation and controls the enabling status of the L0TileInfo configuration switch.

## Prototype

```python
set_conv_tile_shapes(tile_l1_info: pypto.TileL1Info, tile_l0_info: pypto.TileL0Info = None) -> None
```

## Parameters

| Parameter      | Input/Output| Description                                  |
|--------------|-----------|--------------------------------------|
| tile_l1_info | Input     | TileShape configuration information for convolution computation at the L1 buffer level.|
| tile_l0_info | Input     | TileShape configuration information for convolution computation at the L0 buffer level.|

## Return Value

void

## Constraints

TileShape must meet the following constraints:

**Note: Both the L1 Buffer and L0 Buffer have tileN, but they have different meanings. tileL1Info.tileN indicates the number of output channels, and tileL0Info.tileN indicates the value of n at the L0 level.**

- Alignment constraints:

    - The value of each dimension in tileL1Info must satisfy the following constraints:

        - 1 <= tileHin <= Hin (Hin is the actual height of the input feature map.)

        - When wout % 16 = 0, 1 <= tileHout <= Hout (Hout is the actual height of the output feature map.)

        - When wout % 16!= 0, tileHout = 1

        - 1 <= tileWin <= Win (Win is the actual width of the input feature map.)

        - 1 <= tileWout <= CeilAlign(Wout, 16) (Wout is the actual width of the output feature map.) tileWout must be 16-byte aligned, which means `tileWout % 16 == 0`.

        - C0 <= tileCinFmap <= CeilAlign(Cin, C0) (Cin is the actual number of input feature map channels.)

        - tileCinFmap * sizeof(dtype) % 32 == 0

        - CeilAlign(Cin, C0) % tileCinFmap == 0 (This constraint applies only to Conv3D.)

        - C0 <= tileCinWeight <= CeilAlign(Cin, C0) (Cin is the actual channel count of the weight input.)

        - tileCinWeight * sizeof(dtype) % 32 == 0

        - CeilAlign(Cin, C0) % tileCinWeight == 0 (This constraint applies only to Conv3D.)

        - 1 <= tileN <= CeilAlign(Cout // groups, 16) (Cout is the actual number of channels of the output feature map.)

        - tileN % 16 == 0

        - tileBatch = 1 (number of batches)

    - The value of each dimension in tileL0Info must satisfy the alignment constraints:

        - tileK must satisfy: `C0 <= tileK <= min(kAL1, kBL1)`

        - tileK must satisfy: `tileK % C0 == 0`

        - tileK must satisfy: `kAL1 % tilek == 0`

        - tileK must satisfy: `kBL1 % tilek == 0`

        - tileW must be 16-element aligned, that is, `tileW % 16 == 0`

        - tileW must satisfy: `1 <= tileW <= tileWout`

        - tileH must satisfy: `1 <= tileH <= tileHout`

        - tileN (representing the size of n at the L0 level) must be 16-element aligned, that is, `tileN % 16 == 0`

        - tileN must satisfy: `1 <= tileN <= CeilAlign(tileL1Info.tileN, 16)`

        Where:

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

    - Constraints on the L0 Buffer and L1 Buffer dimensions:

        - 1 <= tileL0Info.tileH <= tileL1Info.tileHout and tileL1Info.tileHout % tileL0Info.tileH == 0

        - 1 <= tileL0Info.tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileL0Info.tileW == 0

        - 1 <= tileL0Info.tileN <= tileL1Info.tileN

- Buffer space constraints:

    - L0A Buffer, L0B Buffer, and L0C Buffer space constraints:

        ```txt
        CeilAlign(tileH * tileW, 16)* CeilAlign(tileK, C0) * sizeof(dtype) <= L0A_size

        CeilAlign(tileK, C0) * CeilAlign(tileN, 16) * sizeof(dtype) <= L0B_size

        CeilAlign(tileH * tileW, 16)* CeilAlign(tileN, 16) * sizeof(FP32) <= L0C_size
        ```

        Where:

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `L0A_size = 65536 Byte`

        - `L0B_size = 65536 Byte`

        - `L0C_size = 131072 Byte`

        - `ALIGN_SIZE_32 = 32`

    - L1 Buffer space constraints:

        ```txt
        CeilAlign(hinL1 * winL1 * tileCinFmap * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(nL1 * kBL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(tileN * sizeof(dtype), ALIGN_SIZE_32) <= L1_size
        ```

        Where:

        - `hinL1 = min((tileHout - 1) * strideH + (Kh - 1) * dilationH + 1, Hin)` (Hin is the height of the input feature map.)

        - `winL1 = min((tileWout - 1) * strideW + (Kw - 1) * dilationW + 1, Win)` (Win is the width of the input feature map.)

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `nL1 = tileN` (number of output channels)

        - `dtype is the data type of input_conv (input matrix).`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

        - `CeilAlign(value, align) {  return ((value + align - 1) // align) * align;}`

- Constraints in special scenarios:

    - If `tileL0Info` is not passed, the default `TileL0Info` instance is automatically used, and the L0TileInfo switch is automatically disabled. If a valid `tileL0Info` is passed, the L0TileInfo switch is automatically enabled.

    - The convolution kernel/channel dimension configuration must match the number of input and output channels and the kernel size of the actual convolution operator to prevent the tile size from exceeding the operator dimension range.

## Examples

```python
# Construct the L1 tile configuration (ensure that each value is within the valid range).
l1_tile = pypto.pypto_impl.TileL1Info(
    tileHin=4,        # 1 <= tileHin <= Hin must be satisfied.
    tileHout=4,       # 1 <= tileHout <= Hout must be satisfied.
    tileWin=8,        # 1 <= tileWin <= Win must be satisfied.
    tileWout=8,       # 1 <= tileWout <= Wout must be satisfied.
    tileCinFmap=16,   # 1 <= tileCinFmap <= Cin must be satisfied.
    tileCinWeight=32, # 1 <= tileCinWeight <= Cin must be satisfied.
    tileN=16,         # 1 <= tileN <= Cout must be satisfied.
    tileBatch=1       # tileBatch = 1 must be satisfied.
)

# Construct the L0 tile configuration (satisfying alignment constraints).
l0_tile = pypto.pypto_impl.TileL0Info(
    tileH=2,   # tileH < = tileL1Info.tileHout and tileL1Info.tileHout % tileH = = 0 must be satisfied.
    tileW=8,   # tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileW == 0 must be satisfied.
    tileK=32,  # tileK % C0 == 0 (C0 is the alignment base number, which is determined by the data type.)
    tileN=16   # tileN % 16 = = 0 must be satisfied.
)

# Setting the convolution TileShape (with L0TileInfo enabled).
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile, tile_l0_info=l0_tile)

# Set only the L1 TileShape (with L0TileInfo disabled).
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile)
```
