# dw（conv_backward_filter）实现设计

前置：dx 已合入（`ConvBp` 命名空间、TileShape 体系、`convbp_dx_operation_impl.cpp` 均存在）。dw 在其基础上新增。

## 1. 接口语义

```
conv_backward_filter(grad_output, input, filter_size, out_dtype, strides, paddings, dilations, *, groups=1)
```

- grad_output (N, Cout, Hout, Wout)、input (N, Cin, Hin, Win)、filter_size (Cout, Cin, Kh, Kw)。
- 输出 = 权重梯度，shape 与 filter_size 一致，**无 dx 的输出范围问题**。
- 但要求 grad_output 的 Hout/Wout 与 input 及卷积属性满足**正向卷积关系**，否则校验报错：

```
Hout = (Hin + padTop + padBottom - dilationH*(Kh-1) - 1) / strideH + 1
Wout = (Win + padLeft + padRight - dilationW*(Kw-1) - 1) / strideW + 1
```

- 首版仅支持 2D（grad_output/input/filter_size 均为 4 维），不支持空 Tensor。

## 2. 入口与校验（ConvBackwardFilter / FilterCheck* 系列）

1. 平台校验 DAV_2201；仅 2D（三输入都 4 维，strides/dilations 长 2、paddings 长 4）。
2. 属性归一化：前补 D 轴（stride/dilation 补 1、padding 补 0）成 3D 语义，与 dx 一致。
3. 校验链（对应 dx 的 BpCheck*，公式见 constraints-and-formulas.md）：
   - **FilterCheckOriginShape**：三组 shape 各维 [1, 1000000]。
   - **FilterCheckOutputShape**：grad_output.Batch == input.Batch；input.Cin == filterSize.Cin × groups；Hout/Wout 满足正向卷积关系。
   - **FilterCheckAttrShape**：padding < dilated kernel；groups 范围；Cout 一致且整除 groups；**FilterCheckLoad3dShape**：Kh/Kw ≤ 255 且 `Kh × Kw × (32B/dtype) ≤ 65535`（注意这里 k0 = Cin0 = 32B/dtype，N 轴内块，与 dx 的 Cout0=16 不同）。
   - **FilterCheckTileTiling**：tileML1 16 对齐；tileNL1 为 Cin0×Kh×Kw 倍数；tileKL1 < Wout 或整除；L0 三 tile 16 对齐且 ≤ CeilAlign(L1,16)；L0A/L0B/L0C 容量；numTileL0 超 MAX_LOOP(5000) 告警。
   - **FilterCheckL1SizeTiling**：L1 容量（公式见 constraints-and-formulas.md §4）。
4. 输出 FZ shape（FilterGetResTensorShape）：`{groups×cin1PerGroup×Kh×Kw, cout1PerGroup, 16, cin0}`，resTensor format = TILEOP_FRACTAL_Z。

## 3. TensorGraph（FilterConstructTensorGraph*）

1. **FilterGetOperandVecIn**：两输入 NZ 视图——grad_output `{batch, groups×cout1PerGroup, ho, wo, 16}`、input `{batch, groups×cin1PerGroup, hi, wi, cin0}`（均 TILEOP_NC1HWC0）。
2. 插两个 OP_FAKE_TRANS（两输入 ND→NZ）。
3. 加 `OP_CONV_BP_FILTER_2D` 节点（NZ 进，FZ resTensor 出）。
4. **FilterGetFinalResTensorFZ2NCHW**：输出 ND tensor 即 filterSize，插 OP_FAKE_TRANS（FZ→ND），挂原始 shape/卷积属性（FilterSetTensorOpAttr，与 dx 同名 key，但 ori_input_shape 替代 ori_weight_shape）。
5. 依赖：`infer_tensor_format.cpp` 的 transdata 支持需扩展 **FRACTAL_Z → ND**。

## 4. TileGraph 循环结构（ConstructDwTileGraph）

```
ConstructDwTileGraph:
  SetDwConvBpAttrParam / SetDwTensorGraphNodes / SetDwConvBpShapeInfo
  for group:
    IterDwL1ExpandFunc:
      for nL1Offset in [0, cin1KdKhKwCin0) step nL1:     // N = Cin×Kd×Kh×Kw×Cin0(展开)
        nL1Size = min(剩余, nL1)
        for mL1Offset in [0, coutPerGroup) step mL1:     // M = Cout
          mL1Size = min(剩余, mL1)
          IterDwL0ExpandFunc:
            for nL0Offset in [0, nL1Size) step nL0:
              for mL0Offset in [0, mL1Size) step mL0:
                IterKExpandFunc:
                  isFirstKOverall = true                 // 跨 batch 的全局首 K
                  for batchDoutOffset in [0, batch×Dout): // Batch 在 K 累加轴内侧
                    batchOffset = batchDoutOffset / Dout
                    for kL1Offset in [0, hout*wout):
                      UpdateDwKL1IterInfo()
                      构造 cL0 resTensor
                      置 aL1/bL1 更新标志
                      for kL0Offset in [kL1Offset, kL1Offset+kL1Size):
                        kL0Size 不跨 wout 行
                        isLastK = (最后 batch) && (kL0 累加到全量 K)
                        isFirstK = isFirstKOverall; isFirstKOverall = false
                        repeatStride=0, repeatTime=1, wStride=CeilAlign(kL0Size,16)
                        ConstructDwGradOutputTile()   // L1(懒加载)+L0A 转置
                        ConstructDwInputTile()        // L1(懒加载)+Load3D
                        DoDwMmad()
                ConstructDwCopyOutTile()   // 全量 K（含全部 batch）累加完才搬出
```

关键差异（相对 dx）：
- **K 轴跨 batch**：isFirstK/isLastK 以 (Batch)HoWo 全量 K 为界；isLastK 需同时判断「当前是最后一个 batchDoutOffset」且「kL0 已推进到 orgHoutWout」。
- **CopyOut 在 mL0 循环内、batch 循环外**。
- `SetDwConvBpShapeInfo` 额外计算：`orgHoutWout = Hout×Wout`、`orgBatchDout = Batch×Dout`、`cin1KdKhKwCin0 = cin1PerGroup×Kd×Kh×Kw×cin0`。

## 5. 区间反算与各 Tile 构造要点

### 5.1 UpdateDwKL1IterInfo（K tile → fmap 有效区间反算）

1. **kL1Size**：kL1 < wout 时取「本行剩余 `wout - kL1Offset % wout`」与 kL1 的 min（不跨行）；否则 `min(hout*wout - kL1Offset, kL1)`。
2. 分解：`houtL1Offset = kL1Offset / wout`、`houtL1Size = CeilDiv(kL1Size, wout)`、`woutL1Offset = kL1Offset % wout`、`woutL1Size = min(kL1Size, wout)`。
3. **反算 fmap（hin/win）区间**（stride/dilation 正向扩展）：`hinL1Offset = houtL1Offset*strideH - padTop`（可为负），`endHinOffset = hinL1Offset + (houtL1Size-1)*strideH + (Kh-1)*dilationH + 1`；与 [0, Hin) 求交集得 hinL1Size（无交集则 0）。win 同理。
4. **只搬有效区间**：L1 上的 fmap 不含 pad 区，pad 留给 im2col 填 0（见 5.3）。

### 5.2 ConstructDwGradOutputL1Tile（OP_L1_COPY_IN_CONV_BP_DW_DY）+ L0A 转置（OP_L1_TO_L0A_CONV_DW）

- L1 目标 shape `{1, cout1L1Size, CeilDiv(houtL1Size*woutL1Size,16), 16, cout0}`——**HoWo 合轴按 16 分型**的 NZ 格式。
- GM 源 offset `{batch, mL1Offset/cout0, houtL1Offset, woutL1Offset, 0}`，shape `{1, cout1L1Size, houtL1Size, woutL1Size, cout0}`。
- L0A 目标 shape `{CeilAlign(mL0Size,16), CeilAlign(kL0Size,16)}`；属性 postM = mL0Offset、postK = kL0Offset - kL1Offset。

### 5.3 ConstructDwInputL1Tile（复用 OP_L1_COPY_IN_CONV）+ Load3D（OP_LOAD3D_CONV_DW）

- L1 目标 shape `{1, cin1L1Size, hinL1Size, winL1Size, cin0}`，其中 `cin1L1Size = CeilDiv(nL1Size/(Kh*Kw), cin0)`。
- GM 源 offset `{batch, nL1Offset/(Kh*Kw*cin0), max(hinL1Offset,0), max(winL1Offset,0), 0}`；只搬有效区间（offset/size 已在 5.1 裁剪到 [0, Hin)×[0, Win)）。
- **im2col 的 pad 按 tile 相对计算**（与正向 load3d 的差异）：
  - `paddingTop = max(0, -hinL1Offset)`（L1 只搬了有效区，负 offset 部分是 pad）
  - `paddingBottom = max(0, endHin - orgHin)`，其中 `endHin = hinL1Offset + (houtL1Size-1)*strideH + dilatedKh`
  - W 轴同理（paddingLeft/paddingRight）
- L0B 目标 shape `{CeilAlign(kL0Size,16), CeilAlign(nL0Size,16)}`；postM = kL0Offset-kL1Offset（注意 dw 中 postM 对应 K 维）、postK = nL0Offset（对应 N 维，与 dx 的 postM/postK 语义轴不同）。
- stride/dilation/filter 全部用真实卷积属性；`repeatStride=0, repeatTime=1, wStride=CeilAlign(kL0Size,16)`。
- 由于 `pad < dilation_k` 的约束，k 轴不存在完全跳过的行，无需考虑跳过清 0。

### 5.4 DoDwMmad

与 dx 的 DoMmad 同构（首 K TILE_A_MUL_B / 其后 TILE_A_MULACC_B、cL0PartialSum 为 DT_FP32 NZ），差异仅：B 侧句柄是 inputTensorPtr；属性 BpSetDwAMulBAttr（nzAttr 用 gradOutput/input/res 三 format 组合）。L0C 上是 FZ 布局 (Cin1, HkWk, Cout1, Cout0, Cin0)。

### 5.5 ConstructDwCopyOutTile（OP_L0C_COPY_OUT_CONV_DW）

- copyOutMode=NZ2NZ、l0cValidMN={mL0Size, nL0Size}、isConv3D 标志。
- GM res 为 FRACTAL_Z (Cin1×HkWk, Cout1, Cout0, Cin0) 视图，按该视图 NZ2NZ 搬出：L0C 的 N 方向(CinHkWk)按 Cin0 分块对应 Cin1×HkWk 轴，M 方向(Cout)按 Cout0 分块对应 Cout1 轴。
- GM 目标 offset：`{dstNOffset/cin0, dstMOffset/cout0, 0, 0}`（dstNOffset = nL1Offset+nL0Offset，dstMOffset = mL1Offset+mL0Offset）。
- 搬出不带激活，固定 NoRelu。

## 6. TileOp 实现意图（conv_bp_pto.h，自行实现函数体）

### TLoadConvBPDwDedy<isConv3D, T, U>

参数：(dst, src, offset0..4, shape0..4)。语义：GM(NC1HWC0) → L1 (1, Cout1, HoWo/16, 16, Cout0)。

实现套路：
1. **先清零**：zeroTile 覆盖 L1 整个 tile 区域（stcDstShape1×stcDstShape2×stcDstShape3 行 × c0 列），TFILLPAD + `pipe_barrier(PIPE_MTE2)`——覆盖 HoWo 尾部不满 16 对齐的脏数据。
2. 逐 Cout1 外层循环：每次搬一个 `pto::ConvTile<Mat, Type, stcC1×stcHw16×stc16×32B, Layout::NDC1HWC0, ConvTileShape<1,1,1,-1,-1,c0Size>>`（一个 Cout1 的 (houtL1Size×woutL1Size, c0) 连续块），gmOffset 按 5 轴 stride 累加，目的偏移 `srcCout1 × dstHw16×16 × 32B`。

### TLoadDwA<T, U>

参数：(dst, src, postM, postK)。语义：L1 (1, Cout1, HoWo/16, 16, Cout0) NZ → L0A (Cout1, HoWo/16, Cout0, 16) ZZ（**只转置、不逆序**，对比 dx 的 TLoad2DDX）。

实现套路：单次 `TEXTRACT<pto::TileLeft<...>, pto::Tile<..., RowMajor, ..., ColMajor>>(l0, l1, postM, postK)`，源 tile 用 (Cout1×Cout0) × (HoWo/16×16) 的行主序分型，目的 tile 为 TileLeft，postM/postK 定位 L0 子块。

### TLoad3DDW<isConv3D, T, U> 与辅助 TImg2ColToL0B<DstTensor, SrcTensor>

语义：L1 fmap (Cin1, HiWi/16, 16, Cin0) → L0B (HoWo/16, Cin1, HkWk, Cin0, 16) img2col。

实现套路：
1. 源/目的均用静态 shape 元组实例化 `select_srcTensor<isConv3D, U, elements, c0Size>` 与 `pto::TileRight`，`SetConvTileParams<isConv3D>` 把 padList/filter/stride/dilation/padValue/repeat 参数写入 src tensor。
2. `TImg2ColToL0B` 直接下发硬件指令 `img2colv2_cbuf_to_cb`，前置三个寄存器配置：
   - `set_fmatrix_b`：fmapW(低16bit) | fmapH(高16bit) | padList[4] 每个 8bit 从 bit32 起
   - `set_l3d_rpt`：repeatStride | repeatTime<<16 | repeatMode<<24
   - `set_padding`：按 dtype 位宽组装 padValue
3. stepM = dst 有效行数、stepK = CeilAlign(有效列, c0Size)；filter 高于 255 时拆 lowFilter + highFilter 标志；注意**指令参数顺序与 TIMG2COL 接口相反**（stepK, stepM, posK, posM）。

### TStoreConvDW<mode, isConv3D, T, U>

参数：(dst, src, offset0..4, realM, realN)。语义：L0C FZ (Cin1, HkWk, Cout1, Cout0, Cin0) → GM FRACTAL_Z，按 `pto::GlobalTensor<Type, Shape<1,-1,-1,16,c0Size>, Stride<1,-1,c0Size*16,c0Size,1>, Layout::NZ>` 视图 NZ2NZ 搬出（fixpipe 语义，NoRelu，复用 TSTORE）。

## 7. 后续优化备忘（首版不做）

- K 轴累加轴（(Batch)HoWo）较大时的部分累加优化：中间结果按 FP32 落 GM 累加，需 vectile 解耦后引入 cast；首版在 L0C 上累加完全量 K，超大 K 可先加限制或日志告警。
