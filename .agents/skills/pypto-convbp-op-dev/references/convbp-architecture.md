# 卷积反向算子架构与 dx/dw 差异对照

## 1. 数学本质：卷积反向 = 两个矩阵乘

PyPTO 在 A2/A3 上把卷积反向实现为 Cube 矩阵乘（MMAD）+ Load3D img2col 展开，M/K/N 轴映射是全部设计的出发点。

### dx（conv_backward_input，输入梯度）

```
     CoutHkWk                        Cin
    ┌─────────┐      ┌───Cin────┐      ┌────Cin───┐
    │         │      │          │      │          │
   HiWi       │  x  CoutHkWk    │  =  HiWi        │
    │         │      │          │      │          │
    └─────────┘      └──────────┘      └──────────┘
         ↑                 ↑                 ↑
   GradOutput             Weight        GradFmap
```

- **M = Hin×Win**（合轴），**K = CeilAlign(Cout,16)×Kh×Kw**，**N = Cin**
- A 矩阵来自 grad_output 的 img2col 展开（Load3D），B 矩阵来自 weight 的转置逆序展开（Load2D）
- Batch 在最外层，每个 batch 独立完成 K 累加并搬出

### dw（conv_backward_filter，权重梯度）

```
    (Batch)HoWo
    ┌─────────────┐      ┌──CinHkWk──┐      ┌─CinHkWk──┐
    │             │      │           │      │          │
   Cout           │  x  HoWo         │  =  Cout        │
    │             │      │           │      │          │
    └─────────────┘      └───────────┘      └──────────┘
         ↑                    ↑                  ↑
    GradOutput             Fmap(load3d)       GradWeight
```

- **M = Cout**，**N = Cin×Kh×Kw**，**K = (Batch)Hout×Wout**（合轴，Batch 在 K 累加轴内侧循环）
- 输出不含 Batch 轴；必须在 L0C 上累加完全量 K（跨所有 batch）后才搬出

## 2. 内存层级数据流

```
GM ──MTE2(TLOAD)──> L1 ──MTE1(TEXTRACT/Load3D/Load2D)──> L0A/L0B ──MMAD──> L0C ──FIX(fixpipe)──> GM
```

| 层级 | 容量(A2/A3) | 用途 |
|------|------------|------|
| L1 | 512KB | GM→L0 的中转，存放 NZ/分型数据 |
| L0A | 64KB | A 矩阵（dx: grad_output 展开；dw: grad_output 转置） |
| L0B | 64KB | B 矩阵（dx: weight 转置逆序；dw: fmap img2col 展开） |
| L0C | 128KB | FP32 累加结果 |

pto-isa 关键限制：**只支持 GM 侧 stride，L1 内数据必须连续排布**。因此所有带间隔/跨块的搬运都退化为「外层循环 + 每次搬运一个最小分型（通常一个 C0=16 元素块）」的最简实现。

## 3. 数据格式与关键常量

| 格式 | 布局 | 用途 |
|------|------|------|
| ND (NCHW) | 用户输入/输出 | Python 接口层 |
| NC1HWC0 (5HD/NZ) | (N, C1, H, W, C0) | grad_output / fmap 的 L1/GM 中间格式 |
| NDC1HWC0 | (N, D, C1, H, W, C0) | 3D 变体，反向内部统一按 3D 语义处理（2D 时 D 补 1） |
| FRACTAL_Z (FZ) | (C1×HkWk, N1, N0, C0) | weight 的 L1/GM 格式；dw 的 GM 输出格式 |
| L0A ZZ / L0B NZ | 矩阵分型 | TileLeft / TileRight |

关键常量：
- `Cin0 = Cout0 = 16`（fp16/bf16 下 32B/dtype；dx 的 K 块 = Cout0=16，dw 的 N 内块 = Cin0=32B/dtype）
- `MKN_M_VALUE = MKN_N_VALUE = 16`（MMAD M/N/K 块大小）
- `ALIGN_SIZE_32 = 32`（字节对齐）、`MAX_PAD_KERNEL = 255`、`MAX_DILATION_STRIDE = 63`、`SHAPE_INNER_AXIS_MAX_SIZE = 65535`（Load3D 内轴上限）

## 4. TensorGraph（格式转换层）

dx 与正向一致：

```
grad_output ──Transdata(ND→NC1HWC0)──┐
                                     dx ──→ grad_fmap(5HD) ──Transdata(5HD→ND)──→ Output
weight ──────Transdata(ND→FZ)───────┘
```

dw 双输入 5HD、输出 FZ：

```
grad_output ──Transdata(ND→NC1HWC0)──┐
                                     dw ──→ grad_weight(FZ) ──Transdata(FZ→ND)──→ Output
input ───────Transdata(ND→NC1HWC0)──┘
```

- 转换用 `OP_FAKE_TRANS` 节点 + `FAKE_TRANS_IN/OUT_FORMAT_ATTR` 属性描述。
- 原始 shape（`ori_gradOutput_shape`/`ori_weight_shape`/`ori_input_shape`/`ori_res_shape`）与卷积属性（groups/paddings/strides/dilations/is_conv3d）挂到算子节点属性上，供 TileGraph 展开与 NZ2NZ 校验使用。
- dw 需要 `framework/src/passes/tensor_graph_pass/infer_tensor_format.cpp` 额外支持 FRACTAL_Z→ND 的 transdata 推断。

## 5. dx / dw 差异对照总表

| 维度 | dx | dw |
|------|-----|-----|
| Python 接口 | `conv_backward_input(grad_output, input_size, weight, out_dtype, strides, paddings, dilations, *, groups=1)` | `conv_backward_filter(grad_output, input, filter_size, out_dtype, strides, paddings, dilations, *, groups=1)` |
| 输出 shape 来源 | 由 `input_size` 传入（有效输出是一个范围） | 由 `filter_size` 唯一确定 |
| M / K / N | HiWi / Cout×HkWk / Cin | Cout / (Batch)HoWo / Cin×HkWk |
| Batch 位置 | 最外层（每 batch 独立 K 累加+搬出） | K 累加轴内侧（全 batch 累加完才搬出） |
| 算子 opcode | `OP_CONV_BP_INPUT_2D` / `OP_CONV_BP_INPUT_3D` | `OP_CONV_BP_FILTER_2D` |
| tile 配置 API | `set/get_convbp_input_tile_shapes` | `set/get_convbp_filter_tile_shapes` |
| tile 配置 key | `convbp_input_tile_shapes` | `convbp_filter_tile_shapes` |
| GM 输出格式 | NC1HWC0（5HD） | FRACTAL_Z |
| 实现文件 | `convbp_dx_operation_impl.cpp` | `convbp_dw_operation_impl.cpp` |
| L1 搬入 | grad_output 前放大（`OP_L1_COPY_IN_CONV_BP_DX_DY`）；weight FZ 直搬（`OP_L1_COPY_IN_CONV_BP`） | grad_output 清零+逐 Cout1（`OP_L1_COPY_IN_CONV_BP_DW_DY`）；fmap 有效区间直搬（复用 `OP_L1_COPY_IN_CONV`） |
| L0A 构造 | `OP_LOAD3D_CONV`（img2col） | `OP_L1_TO_L0A_CONV_DW`（NZ→ZZ 转置） |
| L0B 构造 | `OP_LOAD2DDX_CONV`（FZ 转置+逆序） | `OP_LOAD3D_CONV_DW`（img2col） |
| L0C 搬出 | `OP_L0C_COPY_OUT_CONV`（复用正向，5HD 视图） | `OP_L0C_COPY_OUT_CONV_DW`（FZ 视图 fixpipe） |
| M 轴不跨行约束 | tileML1 < Win 或 Win 的整数倍 | ——（M=Cout 只要 16 对齐） |
| K 轴不跨行约束 | ——（K=CoutHkWk） | tileKL1 < Wout 或 Wout 的整数倍 |

## 6. 循环层级对照

dx（详见 dx-design.md）：

```
for group → for batch → for din
  → L1: for nL1(Cin) → for mL1(HiWi)
    → L0: for nL0 → for mL0
      → for kL1(CoutHkWk) → for kL0 → 搬入/展开/MMAD
      → CopyOut（本 mL0 tile 的 K 已累加完）
```

dw（详见 dw-design.md）：

```
for group
  → L1: for nL1(CinHkWk) → for mL1(Cout)
    → L0: for nL0 → for mL0
      → K: for batch → for kL1(HoWo) → for kL0 → 搬入/展开/MMAD
      → CopyOut（跨全部 batch 的 K 累加完）
```

两者 MMAD 完全同构：首 K 用 `TILE_A_MUL_B`（无累加输入），其后 `TILE_A_MULACC_B` 在 L0C 上 FP32 累加，末 K 输出结果 tensor。
