# dx / dw 文件变更地图

本地图给出两个算子各自需要触达的全部文件与修改要点。命名清单（opcode 名、TileOp 名、Python API 名、配置 key）是唯一事实来源，多模块必须保持一致。

通用约定：
- 新代码统一放 `npu::tile_fwk::ConvBp` 命名空间（框架侧）。
- 操作数下标：dx `INPUT_GRAD_OUTPUT_IDX=0, INPUT_WEIGHT_IDX=1, INPUT_BIAS_IDX=2`；dw `FILTER_GRAD_OUTPUT_IDX=0, FILTER_INPUT_IDX=1`。
- 属性 key 前缀 `op_attr_`：`op_attr_paddings/strides/dilations/groups/bias_flag/is_conv3d/ori_gradOutput_shape/ori_weight_shape/ori_input_shape/ori_res_shape`。

## A. 共享基础模块（dx 首次引入；dw 阶段做拆分改造）

| 文件 | 修改要点 |
|------|---------|
| `framework/include/tilefwk/tilefwk_op.h` | `ConvBp` 命名空间：`ConvBpTileL1Info(tileML1,tileKL1,tileNL1)`、`ConvBpTileL0Info(tileML0,tileKL0,tileNL0)`、`ConvBpExtendParam`（bias 预留）、`ConvBackwardInput(...)`（dw 阶段追加 `ConvBackwardFilter(...)`） |
| `framework/include/tilefwk/tile_shape.h` | `ConvBpTile` 结构体（含 valid()/ToString()）、`TileType::CONV_BP`、`TileShape` 构造函数与成员。dw 阶段：成员拆分为 `convBpInputTile`/`convBpFilterTile`，接口拆为 `Set/GetConvBpInputTile`、`Set/GetConvBpFilterTile` |
| `framework/src/interface/operation/tile_shape.cpp` | `ConvBpTile::valid/ToString` 实现；`SetConvBp*Tile` 写 `ConfigManagerNg`（key：dx 单套阶段为 `convbp_tile_shapes`，dw 拆分后为 `convbp_input_tile_shapes`/`convbp_filter_tile_shapes`）；`ToString(TileType)` 增加分支 |
| `framework/src/interface/configs/config_manager_ng.h` | `GetConvBpTile`（dw 拆分后 `GetConvBpInputTile`/`GetConvBpFilterTile`） |
| `framework/src/interface/configs/config_manager_ng.cpp` | 类型名注册表加 `ConvBpTile`；`HasConfig`/`GenerateTileShape`/序列化 `AddValue` 同步各 key |
| `framework/src/interface/operation/operation.h` / `.cpp` | dx 新增 `OpAttributeKey`：`skipH`/`skipW`（SKIP_H/SKIP_W）、`hwk`、`kL0Size`、`nL0Size`、`k0Idx`、`n0Idx`、`kL0TileSize`（ConvBP_ 前缀字符串）。dw 复用正向 filter/stride/dilation/padding/postM/postK 系列 |
| `framework/src/interface/operation/operation_impl.h` | `ConvBp` 命名空间共享声明：NCHW/NCDHW/NCL/NC1HWC0/NDC1HWC0 下标常量、attr key 字符串、`LoadStoreConvBpOpAttributeKey`、`ConvBpAttrParam`、`ConvBpGraphNodes`（gradOutput/weight|input/bias/res/cL0PartialSum 句柄）、`ConvBpTileInfo`（org 形状+展开量+tile 六元组；dw 阶段追加 `orgHoutWout`/`orgBatchDout`/`cin1KdKhKwCin0`）、`ConvBpIterInfo`（各级 offset/size + aL1/bL1 更新标志 + isFirstK/isLastK + load3d 参数；dw 阶段追加 `batchDoutOffset`）、`ConstructTileGraph` 声明 |
| `framework/src/interface/operation/operation_impl.cpp` | `ExpandOperationInto` 增加 case：`OP_CONV_BP_INPUT_2D/3D → ConvBp::ConstructTileGraph`（dx）、`OP_CONV_BP_FILTER_2D → ConvBp::ConstructDwTileGraph`（dw） |

## B. dx 专属变更

### B1. Opcode 注册链路

| 文件 | 修改要点 |
|------|---------|
| `framework/src/interface/operation/opcode.h` | enum 增加：算子级 `OP_CONV_BP_INPUT_3D`/`OP_CONV_BP_INPUT_2D`；TileOp 级 `OP_L1_COPY_IN_CONV_BP_DX_DY`、`OP_L1_COPY_IN_CONV_BP`、`OP_LOAD2DDX_CONV`。相关集合判断（IsLoad2D/IsCopyIn/L1 搬入白名单等）同步 |
| `framework/src/interface/operation/opcode.cpp` | `RegisterCube()` 注册：<br>• `OP_CONV_BP_INPUT_2D`：AIC、"CONV_BP_INPUT_2D"、calc 指令 "CUBE_CONV_BP_INPUT_2D"、OpCalcType::CONV、DAV_2201 格式 {NC1HWC0, FRACTAL_Z, ND}→{NC1HWC0}（3D 对应 NDC1HWC0/FRACTAL_Z_3D）<br>• `OP_L1_COPY_IN_CONV_BP_DX_DY`：DDR→L1、实现名 `TileOp::TLoadConvBpDxDedy`、PIPE_MTE2、attrs {srcGmConvValidShape, strideH, strideW, skipH, skipW}<br>• `OP_L1_COPY_IN_CONV_BP`：DDR→L1、`TileOp::L1CopyInConvBpNZ`、PIPE_MTE2<br>• `OP_LOAD2DDX_CONV`：L1→L0B、`TileOp::TLoad2DDX`、PIPE_MTE1、attrs {postK, postN, hwk, kL0Size, nL0Size, k0Idx, n0Idx}<br>`SUPPORT_TILETENSOR_OPS` 追加 TLoadConvBpDxDedy/TLoadConvBPNZ/TLoad2DDX |
| `framework/src/interface/operation/op_infer_shape_impl.cpp` | 新增 `L1CopyInConvBpInferFunc`（按 `l1_tile_shape`+`IS_FMAP` 属性推 shape）并注册两个 L1 搬入 opcode；`OP_LOAD2DDX_CONV` 复用 `L1ToL0ConvInferFunc` |
| `framework/src/codegen/codegen_op.cpp` | `OP_SHAPE_FROM_ATTR` 加入 `OP_L1_COPY_IN_CONV_BP_DX_DY`/`OP_L1_COPY_IN_CONV_BP` |
| `framework/src/cost_model/simulation/common/ISA.h` | `SCHED_CORE_PIPE_TYPE` 增加：CUBE_CONV_BP_INPUT_2D/3D→PIPE_CUBE、LOAD2DDX→PIPE_MTE1、L1_COPY_IN_CONV_BP 与 L1_COPY_IN_CONV_BP_DX_DY→PIPE_MTE_IN |

### B2. TileOp（`framework/src/interface/tileop/conv/conv_bp_pto.h`，新建）

三个模板函数（算法详见 dx-design.md §6）：

| TileOp | 对应 opcode | 一句话职责 |
|--------|------------|-----------|
| `TLoadConvBpDxDedy<isConv3D, T, U>` | OP_L1_COPY_IN_CONV_BP_DX_DY | GM(NC1HWC0)→L1，HW 间按 stride 插空隙（前放大），每次 TLOAD 一个 ConvTile |
| `TLoadConvBPNZ<T, U>` | OP_L1_COPY_IN_CONV_BP | GM(FZ)→L1 整块 NZ 直搬 |
| `TLoad2DDX<T, U>` | OP_LOAD2DDX_CONV | L1 FZ→L0B，(C1,HW,N1) 循环 TEXTRACT 转置 N0C0→C0N0，HW 逆序 |

并在 `framework/src/interface/tileop/TileOpImpl.h` include 该头。

### B3. Operation 实现（`framework/src/interface/operation/convbp_operation_impl.cpp`，新建；dw 合入时重命名为 `convbp_dx_operation_impl.cpp`）

1170 行量级，四段结构：校验链 / TensorGraph / TileGraph / 共享工具函数。设计见 dx-design.md。

### B4. Codegen（`framework/src/codegen/npu/codegen_mte_convbp.cpp`，新建）

| Gen 函数 | opcode | 输出语句形态 |
|----------|--------|-------------|
| `GenMemL1ToL0Load2DConvBpDx` | OP_LOAD2DDX_CONV | `tileOpName(kL0Size, nL0Size, hwk, k0Idx, n0Idx)` |
| `GenMemL1CopyInConvBp` | OP_L1_COPY_IN_CONV_BP_DX_DY | `tileOpName<isConv3D>(dst, src, off[5], shape[5], strideH, strideW, skipH, skipW)` |
| `GenMemL1CopyInConvBpNZ` | OP_L1_COPY_IN_CONV_BP | `tileOpName(dst, src, off[4])` |

在 `codegen_op_npu.h` 声明、`codegen_op_npu.cpp` 映射表登记。

### B5. Passes 适配

| 文件 | 修改要点 |
|------|---------|
| `framework/src/passes/tile_graph_pass/graph_constraint/pre_graph/cube_process.cpp` | `AddL1CopyInAttr`：`OP_LOAD2DDX_CONV` 加入 L0B 类分支（设置 L1_COPY_IN_OUTER=K、INNER=N） |
| `framework/src/passes/tile_graph_pass/graph_partition/l1_copy_reuse.cpp` | `CheckOpListValid` 白名单加 `OP_L1_COPY_IN_CONV_BP_DX_DY`/`OP_L1_COPY_IN_CONV_BP` |

### B6. Python / bindings

| 文件 | 修改要点 |
|------|---------|
| `python/pypto/op/convbp.py`（新建） | `conv_backward_input` + `__validate_inputs`/`__validate_shape`/`__validate_data_type_consistency` |
| `python/pypto/op/__init__.py` | `from .convbp import *` |
| `python/pypto/_controller.py` | `set/get_convbp_input_tile_shapes`（`__all__` 同步） |
| `python/pypto/config.py` | `ConvBpTile` 包装类、`_Options`/`options`/`set_options`/`ConfigScope.get_convbp_input_tile_shapes` |
| `python/src/bindings/controller.cpp` | `ConvBpTileL1Info`/`ConvBpTileL0Info` py class（字段 def_readwrite）、`Set/GetConvBpTile`（dw 阶段改名 `Set/GetConvBpInputTile`） |
| `python/src/bindings/operation.cpp` | 绑定 `ConvBackwardInput` |
| `python/pypto/pypto_impl/__init__.pyi` | 同步 stub |

### B7. 测试 / 文档

| 文件 | 修改要点 |
|------|---------|
| `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_mte/test_codegen_dyn_convbp.cpp`（新建） | CodegenTestBase 派生，逐 TileOp 生成指令字符串校验 |
| `framework/tests/ut/operator/src/test_operation_impl.cpp` | 增加 `Test_ConvBp_*` 用例（SetConvBpTile + FUNCTION 内调 ConvBackwardInput） |
| `python/tests/ut/operation/test_convbp_op.py`（新建） | 合法 + 非法输入用例 |
| `python/tests/st/test_convbp.py`（新建） | NPU/SIM + golden 对拍 |
| `docs/zh/api/tensor_api/operation/pypto-conv_backward_input.md`（新建） | API 文档 |
| `docs/zh/api/tensor_api/config/pypto-set|get_convbp_input_tile_shapes.md`（新建） | tile 配置文档 |
| `docs/zh/api/index_hiascend.md`、各 `index.md` | 登记新文档 |
| `docs/zh/guide/appendix/trouble_shooting/conv.md` | 新 opcode 排查行 |

## C. dw 专属变更（dx 合入后的增量）

### C1. TileShape 拆分

`convBpTile` 拆为 `convBpInputTile`/`convBpFilterTile` 两套成员与 Set/Get 接口；配置 key `convbp_tile_shapes` → `convbp_input_tile_shapes` + `convbp_filter_tile_shapes`；python `config.py`/`_controller.py`/bindings 同步拆分（`SetConvBpInputTile`/`SetConvBpFilterTile` 等）。dx 实现文件同步改名 `convbp_operation_impl.cpp` → `convbp_dx_operation_impl.cpp`。

### C2. Opcode 注册链路

| 项 | 内容 |
|----|------|
| 算子级 | `OP_CONV_BP_FILTER_2D`：AIC、"CONV_BP_FILTER_2D"、calc "CUBE_CONV_BP_FILTER_2D"、OpCalcType::CONV、DAV_2201 输入格式 {NC1HWC0, NC1HWC0} → 输出格式 {FRACTAL_Z} |
| TileOp 级 | `OP_L1_COPY_IN_CONV_BP_DW_DY`（DDR→L1，`TileOp::TLoadConvBPDwDedy`，PIPE_MTE2，attrs {srcGmConvValidShape}）；`OP_L1_TO_L0A_CONV_DW`（L1→L0A，`TileOp::TLoadDwA`，PIPE_MTE1，attrs {postM, postK}）；`OP_LOAD3D_CONV_DW`（L1→L0B，`TileOp::TLoad3DDW`，PIPE_MTE1，attrs {postM, postK, filterH, filterW, strideH, strideW, dilationH, dilationW, padding 系列, padValue, repeatStride, repeatTime, wStride}）；`OP_L0C_COPY_OUT_CONV_DW`（L0C→DDR，`TileOp::TStoreConvDW`，PIPE_FIX，attrs {excludeBufferReuse, l0cValidMN}） |
| infer shape | 四个 opcode 分别注册（L1 搬入用 `L1CopyInConvBpInferFunc`，L0A/L0B 类用 `L1ToL0ConvInferFunc`，L0C 用 `L0CCopyOutConvInferFunc`） |
| ISA.h | `CUBE_CONV_BP_FILTER_2D`→PIPE_CUBE |
| opcode 集合 | 新 opcode 加入 `IsCopyIn`/`OP_SHAPE_FROM_ATTR`/L1 搬入白名单/`operation.h` L0C 搬出集合等 |

### C3. TileOp（追加到 `conv_bp_pto.h`）

| TileOp | 对应 opcode | 一句话职责 |
|--------|------------|-----------|
| `TLoadConvBPDwDedy<isConv3D, T, U>` | OP_L1_COPY_IN_CONV_BP_DW_DY | GM→L1(1, Cout1, HoWo/16, 16, Cout0)：TFILLPAD 清零 + 逐 Cout1 TLOAD |
| `TLoadDwA<T, U>` | OP_L1_TO_L0A_CONV_DW | L1 NZ→L0A ZZ：TEXTRACT 转置（Cout1, HoWo/16, 16, Cout0)→(Cout1, HoWo/16, Cout0, 16)，postM/postK 控偏移 |
| `TLoad3DDW<isConv3D, T, U>`（+辅助 `TImg2ColToL0B`） | OP_LOAD3D_CONV_DW | L1 fmap→L0B img2col：配置 fmatrix_b/l3d_rpt/padding 寄存器后下发 `img2colv2_cbuf_to_cb` |
| `TStoreConvDW<mode, isConv3D, T, U>` | OP_L0C_COPY_OUT_CONV_DW | L0C FZ(Cin1,HkWk,Cout1,Cout0,Cin0)→GM FRACTAL_Z，NZ2NZ fixpipe，NoRelu |

### C4. Operation 实现（`framework/src/interface/operation/convbp_dw_operation_impl.cpp`，新建）

878 行量级。入口 `ConvBackwardFilter`；TileGraph 展开入口 `ConvBp::ConstructDwTileGraph`。设计见 dw-design.md。

### C5. Codegen（追加到 `codegen_mte_convbp.cpp`）

| Gen 函数 | opcode | 备注 |
|----------|--------|------|
| `GenMemL1CopyInConvBpDwDy` | OP_L1_COPY_IN_CONV_BP_DW_DY | 清零参数 + 5 维 offset/shape |
| `GenMemL1ToL0AConvDw` | OP_L1_TO_L0A_CONV_DW | postM/postK |
| （复用）`GenMemL1ToL0Load3D` | OP_LOAD3D_CONV_DW | 与正向 LOAD3D 同语句形态 |
| `GenMemL0CCopyOutConvBpDw`（+`BuildCopyOutConvBpDwParamList`） | OP_L0C_COPY_OUT_CONV_DW | FZ 视图 fixpipe 参数 |

### C6. Passes 适配

| 文件 | 修改要点 |
|------|---------|
| `framework/src/passes/tensor_graph_pass/infer_tensor_format.cpp` | transdata 支持源格式 FRACTAL_Z → ND（dw 输出） |
| `framework/src/passes/tile_graph_pass/graph_constraint/pre_graph/cube_process.cpp` | `OP_L1_TO_L0A_CONV_DW` 加入 L0A 类分支；`OP_LOAD3D_CONV_DW` 加入 L0B 类分支 |
| `framework/src/passes/tile_graph_pass/graph_partition/l1_copy_reuse.cpp` | 白名单加 `OP_L1_COPY_IN_CONV_BP_DW_DY` |

### C7. Python / bindings / 测试 / 文档

- `python/pypto/op/convbp.py` 追加 `conv_backward_filter` + `__validate_filter_inputs`/`__validate_filter_type_consistency`。
- `_controller.py`/`config.py`/`controller.cpp`/`operation.cpp` 按 C1 拆分并新增 filter 侧 API；`set_vec_tile_shapes` 必须在调用 dw 前设置。
- 测试：`framework/tests/ut/interface/src/operation/test_operation_impl_conv.cpp` 增 dw 用例；`test_codegen_dyn_convbp.cpp` 增 4 个 TileOp 用例；`test_convbp_op.py`/`test_convbp.py` 增 dw 用例（basic/stride/batch/dilation/no_pad + 非法输入）。
- 文档：`pypto-conv_backward_filter.md`、`pypto-set|get_convbp_filter_tile_shapes.md`、index 登记、trouble_shooting/conv.md 补 dw 行。
