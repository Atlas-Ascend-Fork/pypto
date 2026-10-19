# dx（conv_backward_input）实现设计

## 1. 接口语义

```
conv_backward_input(grad_output, input_size, weight, out_dtype, strides, paddings, dilations, *, groups=1)
```

- grad_output (N, Cout, Hout, Wout)、weight (Cout, Cin, Kh, Kw)，输出 (N, Cin, Hin, Win) = input_size。
- **输出是范围而非唯一值**（stride 除法信息丢失），每个轴有效范围：

```
HinMin = (Hout - 1) * strideH - padTop - padBottom + (Kh - 1) * dilationH + 1
WinMin = (Wout - 1) * strideW - padLeft - padRight + (Kw - 1) * dilationW + 1
HinMax = HinMin + strideH - 1 ;  WinMax = WinMin + strideW - 1
```

使用者把目标 shape 传入 input_size；框架校验 grad_output 与 input_size 满足上述反向卷积关系（BpCheckOutputShape 逐轴 CHECK）。

## 2. 入口与属性归一化（ConvBackwardInput）

1. 平台校验：`GetNPUArch() == DAV_2201`。
2. 维度校验：grad_output/weight/input_size 同维（3/4/5），strides/dilations 长度=维数-2、paddings=2×(维数-2)。
3. **属性归一化为 3D 语义**：不足 5 维时在前面补 D 轴（stride/dilation 补 1、padding 补 0），置 `isConv1D`/`isConv3D` 标志；1D 需先插 OP_RESHAPE 升 4 维（输出再 reshape 回）。
4. 构造 `ConvBpAttrParam`（paddings/strides/dilations/groups/hasBias + 各 ori shape）。
5. 校验链（见 §3）→ 计算输出 5HD shape → 构造 resTensor（NC1HWC0，3D 为 NDC1HWC0）→ TensorGraph。

输出 5HD shape：`{batch, cin1, hi, wi, cin0}`，其中 `cin0 = 32B/dtype`、`cin1 = groups × CeilDiv(cin/groups, cin0)`。

## 3. 校验链（BpCheck* 系列函数）

按顺序实现，每个函数职责单一：

1. **BpCheckOriginShape**：grad_output/weight/input_size 各维在 [1, 1000000]；bias shape 与 Cin 一致。
2. **BpCheckOutputShape**：input_size 与 grad_output 满足反向卷积关系（逐 H/W/D 轴）。
3. **BpCheckAttrShape**：padding < dilated kernel（每轴双向）；groups 范围；grad_output Cout == weight Cout 且整除 groups；Load3D 上限（Kh/Kw ≤ 255，Kh×Kw×16 ≤ 65535）；**Cout 每组必须是 c0(32B/dtype) 的整数倍**（transdata 无法消除输出多余 pad）。
4. **BpCheckTileTiling + BpCheckL0TileTiling**：tile 对齐/范围约束 + L0A/L0B/L0C 容量 + 编译循环量告警（公式见 constraints-and-formulas.md §3）。
5. **BpCheckL1SizeTiling**：L1 容量（公式见 constraints-and-formulas.md §4）。

## 4. TensorGraph（BpConstructTensorGraph*）

1. 1D 时插 OP_RESHAPE（N,C,L → N,C,1,L），bias 插 reshape 变 2 维（首版无 bias）。
2. **BpGetOperandVecIn**：构造 NZ 视图 operand——grad_output NZ shape `{batch, groups×cout1PerGroup, ho, wo, 16}`（TILEOP_NC1HWC0）；weight FZ shape `{groups×cin1PerGroup×kh×kw, cout1PerGroup, 16, cin0}`（TILEOP_FRACTAL_Z；3D 对应 NDC1HWC0/FRACTAL_Z_3D）。
3. 插两个 `OP_FAKE_TRANS`（grad_output ND→NZ、weight ND→FZ），属性带 IN/OUT format。
4. 加 `OP_CONV_BP_INPUT_2D/3D` 节点（NZ operand 进，resTensor 出）。
5. **BpGetFinalResTensorNZ2NZ**：由 5HD res 反推 ND 输出 tensor（Cin = weight Cin × groups），插 `OP_FAKE_TRANS`（5HD→ND），原始 shape 属性挂节点（BpSetTensorOpAttr：groups/paddings/strides/dilations/is_conv3d/ori 三 shape/dynamicResValidShape）。

## 5. TileGraph 循环结构

```
ConstructTileGraph:
  SetConvBpAttrParam(op)          // 从节点属性恢复 ConvBpAttrParam
  SetTensorGraphNodes()           // operand → gradOutput/weight/res 句柄
  SetConvBpShapeInfo()            // org 形状 + 展开量 + tile 六元组 → ConvBpTileInfo
  for group → for batch → for din:
    IterL1ExpandFunc:
      for nL1Offset in [0, cinPerGroup) step nL1:      // N = Cin
        for mL1Offset in [0, hin*win):                 // M = HiWi
          UpdateL1IterInfo()
          if houtL1Size == 0 || woutL1Size == 0: continue   // 纯 pad 区，跳过
          IterL0ExpandFunc:
            for nL0Offset in [0, nL1Size) step nL0:
              for mL0Offset in [0, mL1Size):            // 步进 mL0Size（不跨 win 行）
                构造 cL0 resTensor
                for kL1Offset in [0, kPerGroup) step kL1:   // K = CeilAlign(cout,16)*kh*kw
                  置 aL1/bL1 更新标志
                  for kL0Offset in [kL1Offset, kL1Offset+kL1Size) step kL0:
                    isFirstK = (kL0Offset == 0)
                    isLastK = (kL0Offset + kL0 >= kPerGroup)
                    ConstructGradOutputTile()   // L1(懒加载)+Load3D
                    ConstructWeightTile()       // L1(懒加载)+Load2D
                    DoMmad()
                ConstructCopyOutTile()          // K 累加完，L0C→GM
```

**L1 懒加载**：aL1/bL1 数据只依赖 kL1/nL1/mL1 循环，kL0 内循环不重搬——用 `aL1UpadateFlag`/`bL1UpadateFlag` 标志在 L1 tile 构造函数里自清。

## 6. 各 Tile 构造函数设计要点

### 6.1 SetConvBpShapeInfo 中的展开量（TileGraph 的核心常量）

```
expandHout = (Hout - 1) * strideH + 1            // grad_output 的"前放大"目标尺寸
expandWout = (Wout - 1) * strideW + 1
bpPadUp    = (Kh - 1) * dilationH - padTop       // dx 反向等效 pad（可为负）
bpPadLeft  = (Kw - 1) * dilationW - padLeft
kPerGroup  = CeilAlign(coutPerGroup, 16) * Kh * Kw
cin0 = 16(对齐块) ; cout0 = 16(MMAD K 块)
```

### 6.2 UpdateL1IterInfo（M tile → grad_output 有效区间的反算）

1. **mL1Size**：mL1 < win 时取「本行剩余 `win - mL1Offset % win`」与 mL1 的 min（不跨行）；否则 `min(hin*win - mL1Offset, mL1)`。
2. hin/win 区间：`hinL1Offset = mL1Offset / win`，`hinL1Size = CeilDiv(mL1Size, win)`，`winL1Offset = mL1Offset % win`，`winL1Size = min(mL1Size, win)`。
3. **反算 grad_output（hout/wout）区间**：`houtL1Offset = hinL1Offset - bpPadUp`（可为负，负表示落入 pad 区），`endHoutOffset = houtL1Offset + hinL1Size + (Kh-1)*dilationH`；与 [0, expandHout) 求交集得 `houtL1Size`（无交集则 0，外层跳过）。W 轴同理。
4. `nL1Size = min(cin1PerGroup*cin0 - nL1Offset, nL1)`。

### 6.3 ConstructGradOutputL1Tile（OP_L1_COPY_IN_CONV_BP_DX_DY）

- L1 目标 shape `{1, cout1L1Size, houtL1Size, woutL1Size, cout0}`，其中 `cout1L1Size = (kL1Size/16)/(Kh*Kw)`。
- GM 源区间（5HD 视图）：`srcGmOffset = {batch, group×cout1PerGroup + kL1Offset/16/(Kh*Kw), CeilDiv(max(houtL1Offset,0), strideH), CeilDiv(max(woutL1Offset,0), strideW), 0}`。
- **skip 计算**：houtL1Offset 非 stride 整数倍时，L1 目标起始行前需空 `hL1Skip = CeilAlign(realHoutOffset, strideH) - realHoutOffset` 个位置；`hLoadToL1 = CeilDiv(houtL1Size - hL1Skip, strideH)` 为实际搬运行数（W 同理）。skipH/skipW/strideH/strideW 四个属性传给 TileOp 做前放大。
- `CopyOpAttribute`：`Specified(srcGmOffset), MEM_L1, Specified(srcGmShape), 源 dynRawShape, dstShape`；另设 `srcGmConvValidShape`/`l1_tile_shape`/copyInMode=NZ2NZ/isConv3D。

### 6.4 ConstructWeightL1Tile（OP_L1_COPY_IN_CONV_BP）

- L1 目标 shape `{(nL1Size/cin0)*Kh*Kw, kL1Size/16/KhKw, 16, cin0}`（FZ 子块）。
- GM offset `{(dkOffset*cin1 + cin1Offset)*KhKw, cout1Offset, 0, 0}`，`isConv=true, isFmap=false`。

### 6.5 SetImg2ColAttr（OP_LOAD3D_CONV，L1→L0A img2col）

- stride 固定 1（前放大已在 L1 完成），dilation 用真实值，filterH/W 用真实 Kh/Kw。
- **pad 相对 tile 计算**：`paddingTop = max(0, -houtL1Offset)`；`paddingBottom = max(0, (houtL1Offset + hinL1Size - 1 + dilatedKh) - expandHout)`；W 同理。
- `postM = mL0Offset`，`postK = kL0Offset - kL1Offset`（L0 tile 在 L1 大块中的偏移）。
- load3dv2 参数：`repeatTime=1`、`wStride = CeilAlign(mL0Size, 16)`、repeatStride 沿用、padValue=0、isConv=true、isConv3D=false（反向统一 3D 格式）。

### 6.6 ConstructWeightTile 的 L0 展开属性（OP_LOAD2DDX_CONV）

- L0 目标 shape `{kL0Size, CeilAlign(nL0Size, 16)}`。
- 属性：postK = kL0Offset-kL1Offset、postN = nL0Offset、kL0Size/nL0Size、hwk = Kh×Kw、k0Idx = kL0Offset-kL1Offset、n0Idx = nL0Offset。

### 6.7 DoMmad（TILE_A_MUL_B / TILE_A_MULACC_B）

- 首 K：输入 {A, B(, bias)}；非首 K：输入 {A, B, cL0PartialSum}（DT_FP32，NZ）。
- 末 K 输出 resTensor，否则输出新建 cL0PartialSum `{CeilAlign(mL0Size,16), CeilAlign(nL0Size,16)}`。
- 属性（BpSetAMulBAttr）：isConv、MATMUL_NZ_ATTR（三 tensor format 位组合）、A_MUL_B_ACT_M/K/N = mL0/kL0/nL0、首 K 带 bias 标志。

### 6.8 ConstructCopyOutTile（OP_L0C_COPY_OUT_CONV）

- `cutW = realCutW = mL0Size`（mL0 内 hiwi 连续，逐行搬出）。
- copyOutMode=NZ2NZ；`l0cValidMN = {mL0Size, nL0Size}`。
- GM 目标 offset（5HD res 视图）：`{batch, (nL1Offset+nL0Offset)/cin0, (mL1Offset+mL0Offset)/win, (mL1Offset+mL0Offset)%win, 0}`（3D 插 dinOffset）。
- CopyOpAttribute：`MEM_L0C, Specified(dstResGmOffset), res raw/dynRawShape, dstCL0DynValidShape`。

## 7. TileOp 实现意图（conv_bp_pto.h，自行实现函数体）

### TLoadConvBpDxDedy<isConv3D, T, U>

参数：(dst, src, offset0..4, shape0..4, strideH, strideW, skipH, skipW)。语义：把 GM 上 NC1HWC0 的一块稀疏采样（stride 采样）搬成 L1 上的稠密 houtL1×woutL1（**前放大**）。

实现套路：
1. 从 dst 静态 shape 取 stcDstH/stcDstW，从 src 取 5 个 GM stride（2D 时 D 轴 stride 置 0，之后按 3D 统一寻址）。
2. `zeroTile` 覆盖整行（stcDstH×stcDstW 上取整 16）TFILLPAD 清零（覆盖纯 pad 行），`pipe_barrier(PIPE_MTE2)`。
3. 三层循环 (srcCout1, srcH, srcW)：每次构造一个 `pto::ConvTile<Mat, Type, stcDstW×32B, Layout::NDC1HWC0, ConvTileShape<1,1,1,1,1,c0Size>>`，TASSIGN 到 `dst.GetAddr() + (skipH*dstW + skipW + srcCout1*dstH*dstW + strideH*dstW*srcH + strideW*srcW)*c0Size*sizeof(Type)`，gmOffset 按 5 个 stride 累加，TLOAD 一个 c0 块。

### TLoadConvBPNZ<T, U>

参数：(dst, src, offset0..3)。语义：FZ GM → L1 整块直搬。用 `pto::GlobalTensor<Type, Shape<1,-1,-1,16,c0Size>, Stride<1,-1,c0Size*16,c0Size,1>, Layout::NZ>` 视图，一次 TLOAD 完成 [blockCol, blockRow, fractalRow, 16, c0] 分型。

### TLoad2DDX<T, U>

参数：(dst, src, kL0Size, nL0Size, hwk, k0Idx, n0Idx)。语义：L1 FZ [c1hw, n1, n0=16, c0] → L0B [K, N]，内分型转置（N0C0→C0N0）且 **HW 轴逆序**（dx 的 B 矩阵需要 K 轴按 -stride 语义排布）。

实现套路：
1. `kCout0Num = k0Idx/16`（K 方向第几个 [16,c0] 小块）、`kRepeat = kL0Size/16`、`nCin0Num = n0Idx/c0`、`nRepeat = nL0Size/c0`。
2. 双层循环 (j∈nRepeat, i∈kRepeat)：`iRev = kCout0Num + i`，`hwRevIdx = iRev % hwk`，`hwSrcIdx = hwk - hwRevIdx - 1`（**逆序**），`n1Idx = iRev / hwk`；源偏移 `(nCin0Num+j)*16*c0*n1*hwk + hwSrcIdx*16*c0*n1 + n1Idx*16*c0`，目的偏移 `i*16*nL0 + j*16*c0`。
3. 每次用一个 `pto::Tile<Mat, Type, 16, c0, ColMajor,...>` 源 + `pto::TileRight<Type, 16, c0, -1, -1>` 目的，`TEXTRACT<dstL0, srcL1>(l0, l1, 0, 0)` 转置搬运。

## 8. 设计取舍备忘

- M 轴「不跨 win 行」切分是为了让 L0A 的 M 维数据在 L1 上连续、Load3D 可用 mPos/mExtension 控制偏移，简化实现。
- KL1 必须为 16×Kh×Kw 的倍数：保证一个 L1 K-tile 覆盖完整卷积核块，Load2D 的逆序寻址才能按 (n1, hw) 分解。
- HkWk 在 L1 全载（不考虑超大卷积核），受 Load3D `Kh×Kw×16 ≤ 65535` 限制。
