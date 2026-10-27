/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation_impl.h
 * \brief
 */

#pragma once
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "interface/configs/config_manager.h"
#include "interface/operation/attribute.h"
#include "opcode.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tile_shape.h"

namespace npu::tile_fwk {
class Function;
class Operation;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;

void ExpandOperationInto(Function& function, const TileShape& tileShape, Opcode opCode,
                         const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                         const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);

void TiledAssemble(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                   const LogicalTensorPtr& result, std::shared_ptr<AssembleOpAttribute> attr);

void TiledViewOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                        const LogicalTensorPtr& result, const Operation& op);

namespace Matrix {
const size_t M_INDEX = 0;
const size_t K_INDEX = 1;
const size_t N_INDEX = 2;
const int32_t MATRIX_MAXSIZE = 3;

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string A_MUL_B_TRANS_A = OP_ATTR_PREFIX + "trans_a";
const std::string A_MUL_B_SCALE_A_COPY_IN_MODE = OP_ATTR_PREFIX + "scale_a_copy_in_mode";
const std::string A_MUL_B_TRANS_B = OP_ATTR_PREFIX + "trans_b";
const std::string A_MUL_B_SCALE_B_COPY_IN_MODE = OP_ATTR_PREFIX + "scale_b_copy_in_mode";
const std::string A_MUL_B_GM_ACC = OP_ATTR_PREFIX + "gm_acc";
const std::string A_MUL_B_MX_ATTR = OP_ATTR_PREFIX + "is_mx";
const std::string COPY_IN_L1_PADDING_MODE = OP_ATTR_PREFIX + "copy_in_l1_padding_mode";
const std::string COPY_IN_L1_K_INDEX = OP_ATTR_PREFIX + "copy_in_l1_k_index";
const std::string L1_TO_L0_TRANSPOSE = OP_ATTR_PREFIX + "l1_to_l0_transpose";
const std::string A_MUL_B_BIAS_ATTR = OP_ATTR_PREFIX + "has_bias";
const std::string A_MUL_B_SCALE_ATTR = OP_ATTR_PREFIX + "scale_value";
// relu type 0: NoReLu, 1: ReLu
const std::string A_MUL_B_RELU_ATTR = OP_ATTR_PREFIX + "relu_type";
const std::string A_MUL_B_TRANS_MODE_ATTR = OP_ATTR_PREFIX + "trans_mode";
const std::string A_MUL_B_VECTOR_QUANT_FLAG = OP_ATTR_PREFIX + "vector_quant_flag";

struct MatmulTensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
    NodeType nodeType;
    TileOpFormat format;
    MemoryType memType;
    bool transFlag;

    MatmulTensorInfo(const std::string& nameIn, DataType dtypeIn, const std::vector<int64_t>& shapeIn,
                     const std::vector<int64_t>& offsetIn, NodeType nodeTypeIn, TileOpFormat formatIn,
                     MemoryType memTypeIn, bool transFlagIn = false)
        : name(nameIn),
          dtype(dtypeIn),
          shape(shapeIn),
          offset(offsetIn),
          nodeType(nodeTypeIn),
          format(formatIn),
          memType(memTypeIn),
          transFlag(transFlagIn)
    {}
};

struct MatmulTileInfo {
    int64_t mView = 0;
    int64_t kView = 0;
    int64_t nView = 0;
    int64_t tileML1 = 0;
    int64_t tileML0 = 0;
    int64_t tileNL1 = 0;
    int64_t tileNL0 = 0;
    int64_t tileKL0 = 0;
    int64_t tileKAL1 = 0;
    int64_t tileKBL1 = 0;
};

struct MatmulIterInfo {
    int64_t mOffset = 0;
    int64_t nOffset = 0;
    int64_t kOffset = 0;
    int64_t mL1Size = 0;
    int64_t mL0Size = 0;
    int64_t nL1Size = 0;
    int64_t nL0Size = 0;
    int64_t kAL1Size = 0;
    int64_t kBL1Size = 0;
    int64_t kL0Size = 0;
    bool isFirstK = false;
    bool isLastK = false;
};

struct MatmulGraphNodes {
    LogicalTensorPtr aTensorPtr = nullptr;
    LogicalTensorPtr aScaleTensorPtr = nullptr;
    LogicalTensorPtr bTensorPtr = nullptr;
    LogicalTensorPtr bScaleTensorPtr = nullptr;
    LogicalTensorPtr gmAccumulationTensorPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr scaleTensorPtr = nullptr;
    LogicalTensorPtr cL0PartialSumPtr = nullptr;
    LogicalTensorPtr outTensorPtr = nullptr;

    MatmulGraphNodes() = default;

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr bTensorIn)
        : aTensorPtr(aTensorIn), bTensorPtr(bTensorIn) {};

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr bTensorIn, LogicalTensorPtr gmAccumulationTensorIn)
        : aTensorPtr(aTensorIn), bTensorPtr(bTensorIn), gmAccumulationTensorPtr(gmAccumulationTensorIn) {};

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr aScaleTensorIn, LogicalTensorPtr bTensorIn,
                     LogicalTensorPtr bScaleTensorIn)
        : aTensorPtr(aTensorIn),
          aScaleTensorPtr(aScaleTensorIn),
          bTensorPtr(bTensorIn),
          bScaleTensorPtr(bScaleTensorIn) {};
};

struct MatmulAttrParam {
    int64_t mValue = 0;
    int64_t kValue = 0;
    int64_t nValue = 0;
    int64_t reluType = 0;
    int64_t transMode = 0;
    uint64_t scaleValue = 0;
    bool hasBias = false;
    bool hasScale = false;
    bool transA = false;
    bool transAScale = false;
    bool transB = false;
    bool transBScale = false;
    bool hasMXScale = false;
    bool gmAccumulationFlag = false;
    bool isCMatrixNZ = false;
    bool isGemv = false;

    MatmulAttrParam() = default;

    MatmulAttrParam(bool isATrans, bool isBTrans, bool cMatrixNZ)
    {
        transA = isATrans;
        transB = isBTrans;
        isCMatrixNZ = cMatrixNZ;
    }

    MatmulAttrParam(bool isATrans, bool isAScaleTrans, bool isBTrans, bool isBScaleTrans, bool cMatrixNZ)
        : transA(isATrans),
          transAScale(isAScaleTrans),
          transB(isBTrans),
          transBScale(isBScaleTrans),
          isCMatrixNZ(cMatrixNZ)
    {
        hasMXScale = true;
    }
};

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op);
// kirin9050 matmul 宏指令走独立构图入口（A 驻留 L1，无 L0A 搬运；quant_pre 走 cbuf tile 链路）
void ConstructTileGraphMacroInstr(Function& function, const TileShape& tileShape,
                                  const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                                  const Operation& op);
} // namespace Matrix

namespace Conv {
constexpr const int NCL_N_IDX = 0;
constexpr const int NCL_C_IDX = 1;
constexpr const int NCL_L_IDX = 2;
constexpr const int NCHW_N_IDX = 0;
constexpr const int NCHW_C_IDX = 1;
constexpr const int NCHW_H_IDX = 2;
constexpr const int NCHW_W_IDX = 3;
constexpr const int NCDHW_N_IDX = 0;
constexpr const int NCDHW_C_IDX = 1;
constexpr const int NCDHW_D_IDX = 2;
constexpr const int NCDHW_H_IDX = 3;
constexpr const int NCDHW_W_IDX = 4;
constexpr const int NC1HWC0_N_IDX = 0;
constexpr const int NC1HWC0_C1_IDX = 1;
constexpr const int NC1HWC0_H_IDX = 2;
constexpr const int NC1HWC0_W_IDX = 3;
constexpr const int NC1HWC0_C0_IDX = 4;
constexpr const int NDC1HWC0_N_IDX = 0;
constexpr const int NDC1HWC0_D_IDX = 1;
constexpr const int NDC1HWC0_C1_IDX = 2;
constexpr const int NDC1HWC0_H_IDX = 3;
constexpr const int NDC1HWC0_W_IDX = 4;
constexpr const int NDC1HWC0_C0_IDX = 5;
constexpr const int FRACTALZ_CO1_IDX = 1;
constexpr const int INPUT_FMAP_IDX = 0;
constexpr const int INPUT_WEIGHT_IDX = 1;
constexpr const int INPUT_BIAS_IDX = 2;
constexpr const int CONV1D_INPUT_DIM = 3;
constexpr const int CONV2D_INPUT_DIM = 4;
constexpr const int CONV3D_INPUT_DIM = 5;
constexpr const int MKN_N_VALUE = 16;
constexpr const int MKN_M_VALUE = 16;
constexpr uint32_t PAD_TOP_INDEX = 0;
constexpr uint32_t PAD_BOTTOM_INDEX = 1;
constexpr uint32_t PAD_LEFT_INDEX = 2;
constexpr uint32_t PAD_RIGHT_INDEX = 3;
constexpr uint32_t PAD_HEAD_INDEX = 4;
constexpr uint32_t PAD_TAIL_INDEX = 5;
constexpr uint32_t PAD_STRIDE_H = 0;
constexpr uint32_t PAD_STRIDE_W = 1;
constexpr uint32_t PAD_STRIDE_D = 2;
constexpr int MAX_LOOP = 5000;

const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string CONV_PADDINGS_ATTR = OP_ATTR_PREFIX + "paddings";
const std::string CONV_DILATIONS_ATTR = OP_ATTR_PREFIX + "dilations";
const std::string CONV_STRIDES_ATTR = OP_ATTR_PREFIX + "strides";
const std::string CONV_ORI_FMAP_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_fmap_shape";
const std::string CONV_ORI_WEIGHT_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_weight_shape";
const std::string CONV_ORI_RES_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_res_shape";
const std::string CONV_BIAS_ATTR = OP_ATTR_PREFIX + "bias_flag";
const std::string CONV_3D_FLAG = OP_ATTR_PREFIX + "is_conv3d";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string A_MUL_B_BIAS_ATTR = OP_ATTR_PREFIX + "has_bias";
const std::string CONV_RELU_ATTR = OP_ATTR_PREFIX + "relu_type";
const std::string CONV_SYM_PADDINGS_ATTR = OP_ATTR_PREFIX + "sym_paddings";
const std::vector<int64_t> CONV2D_ATTR_DEFAULT_LIST = {1, 1};
const std::vector<int64_t> CONV3D_ATTR_DEFAULT_LIST = {1, 1, 1};
const std::vector<int64_t> CONV2D_PAD_ATTR_DEFAULT_LIST = {0, 0, 0, 0};
const std::vector<int64_t> CONV3D_PAD_ATTR_DEFAULT_LIST = {0, 0, 0, 0, 0, 0};

class LoadStoreConvOpAttributeKey {
public:
    static const std::string copyInMode;
    static const std::string copyOutMode;
    static const std::string isFmap;
    static const std::string isConv3D;
    static const std::string cutW;     // L0C M方向(hw合轴)的w大小
    static const std::string realCutW; // L0C M方向(hw合轴)的w的validshape
    static const std::string reluType;
};

enum class CopyInMode : int {
    COPY_MOD_INVALID = -1,
    COPY_MOD_ND2ND = 0,
    COPY_MOD_ND2NZ,
    COPY_MOD_NZ2NZ,
    COPY_MOD_DN2NZ
};

enum class CopyOutMode : int {
    COPY_MOD_INVALID = -1,
    COPY_MOD_NZ2ND = 0,
    COPY_MOD_NZ2NZ,
    COPY_MOD_ND2ND,
    COPY_MOD_NZ2DN
};

struct ConvAttrParam {
    std::vector<int64_t> paddings = {0, 0, 0, 0};
    // 非空表示符号(表达式)padding模式: 元素为SymbolicScalar表达式, 由kernel侧按tile计算后传入。
    // 此模式下paddings(int64)统一按0处理, conv内部tile展开采用pad=0语义(view已从正确位置截取)。
    std::vector<SymbolicScalar> symPaddings;
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> oriFmapShape = {0, 0, 0, 0};
    std::vector<int64_t> oriWeightShape = {0, 0, 0, 0};
    std::vector<int64_t> oriResShape = {0, 0, 0, 0};
    std::vector<SymbolicScalar> dynValidResShape;
    int64_t groups = 0;
    int64_t offsetX = 0;
    bool isConv1D = false;
    bool isConv3D = false;
    bool hasBias = false;
    int64_t reluType = 0;
    bool isInOutTensorNZ = false;

    ConvAttrParam() = default;

    ConvAttrParam(std::vector<int64_t> paddingsList, std::vector<int64_t> stridesList,
                  std::vector<int64_t> dilationsList, int64_t groupsValue)
    {
        paddings = paddingsList;
        strides = stridesList;
        dilations = dilationsList;
        groups = groupsValue;
    }
};

struct ConvGraphNodes {
    LogicalTensorPtr fmapTensorPtr = nullptr;
    LogicalTensorPtr weightTensorPtr = nullptr;
    LogicalTensorPtr cL0PartialSumPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr resTensorPtr = nullptr;
};

struct ConvTileInfo {
    int64_t orgBatch = 0;
    int64_t orgCout = 0;
    int64_t orgDin = 0;
    int64_t orgDout = 0;
    int64_t orgHin = 0;
    int64_t orgWin = 0;
    int64_t orgHout = 0;
    int64_t orgWout = 0;
    int64_t orgHoutWout = 0;
    int64_t orgKh = 0;
    int64_t orgKw = 0;
    int64_t orgKd = 0;
    int64_t orgCin = 0;
    int64_t kPerGroup = 0;
    int64_t coutPerGroup = 0;
    int64_t kAL1 = 0;
    int64_t kBL1 = 0;
    int64_t nBL1 = 0;
    int64_t hAL1In = 0;
    int64_t wAL1In = 0;
    int64_t hAL1Out = 0;
    int64_t wAL1Out = 0;
    int64_t kL0 = 0;
    int64_t hL0 = 0;
    int64_t wL0 = 0;
    int64_t nL0 = 0;
    int64_t cin0 = 0;
    SymbolicScalar dynValidBatch = SymbolicScalar(0);
    SymbolicScalar dynValidDout = SymbolicScalar(0);
    SymbolicScalar dynValidCout = SymbolicScalar(0);
    SymbolicScalar dynValidHout = SymbolicScalar(0);
    SymbolicScalar dynValidWout = SymbolicScalar(0);
    SymbolicScalar dynValidHoutL0 = SymbolicScalar(0);
    SymbolicScalar dynValidWoutL0 = SymbolicScalar(0);
    SymbolicScalar dynValidCoutL0 = SymbolicScalar(0);
    SymbolicScalar dynValidBatchL0 = SymbolicScalar(0);
    SymbolicScalar dynValidDoutL0 = SymbolicScalar(0);
};

struct ConvIterInfo {
    int64_t groupOffset = 0;
    int64_t batchOffset = 0;
    int64_t dinL1Offset = 0;
    int64_t doL1Offset = 0;
    int64_t hL1InOffset = 0;
    int64_t hL1OutOffset = 0;
    int64_t wL1InOffset = 0;
    int64_t wL1OutOffset = 0;
    int64_t dkL1Size = 0;
    int64_t dkAL1Size = 0;
    int64_t dkBL1Size = 0;
    int64_t nL1Offset = 0;
    int64_t hL0Offset = 0;
    int64_t wL0Offset = 0;
    int64_t nL0Offset = 0;
    int64_t kAL1Offset = 0;
    int64_t kBL1Offset = 0;
    int64_t kAL1BaseOffset = 0;
    int64_t kBL1BaseOffset = 0;
    int64_t kL0Offset = 0;
    int64_t hinL1Size = 0;
    int64_t winL1Size = 0;
    int64_t houtL1Size = 0;
    int64_t woutL1Size = 0;
    int64_t kAL1Size = 0;
    int64_t kBL1Size = 0;
    int64_t mL0Size = 0;
    int64_t nL1Size = 0;
    int64_t nL0Size = 0;
    int64_t kL0Size = 0;
    int64_t dkBL1SrcOffset = 0;
    bool aL1UpadateFlag = false;
    bool bL1UpadateFlag = false;
    bool isFirstK = false;
    bool isLastK = false;
    // conv3d符号d-pad(表达式)模式: dk裁剪由运行时表达式承载, dkL1Size保持orgKd(kL0循环满额展开),
    // 空block由L1 tile/L0A的valid表达式跳过(load3d不生成K行, mmad按validK=0跳过)。
    // block b(前缀紧排) ↔ 窗口位置 symTmpKdHead + b
    bool hasSymDkPad = false;
    SymbolicScalar symDinL1Offset;   // 当前输出D slice窗口首行(view坐标, 可为负)
    SymbolicScalar symTmpKdHead;     // 窗口头部处于pad的slice数
    SymbolicScalar symDkReal;        // 窗口内真实数据slice数
    SymbolicScalar symDkFirstSrcPos; // 首个真实slice的view坐标 = symDinL1Offset + symTmpKdHead*dilationD
    SymbolicScalar symDkPosClampMax; // 空block src偏移钳制上限 = max(dataD - 1, 0)
    // conv符号h/w-pad(表达式)多L1 tile模式: 逐L1 tile按窗口分解派生表达式,
    // L1 tile窗口起点start = hL1OutOffset*stride与窗宽full = (houtL1Size-1)*stride+dilatedK均为编译期常量,
    // 符号表达式随L1 tile循环展开逐轮更新传递:
    //   padHead   = max(symPad - start, 0)                    该tile头部补零(img2col padTop/Left)
    //   srcOffset = max(start - symPad, 0)                    GM搬运起点(view坐标)
    //   valid     = max(0, min(full - padHead, data - srcOffset))  该tile真实数据量
    //   padTail   = full - valid - padHead(img2col按公式补齐)
    // 单L1 tile时start=0, padHead退化为kernel传入的整窗表达式, 与既有行为一致
    SymbolicScalar symPadTopL1;   // 当前L1 tile头部pad(H轴, 窗口坐标)
    SymbolicScalar symSrcOffsetH; // 当前L1 tile GM搬运H偏移(view坐标, >=0)
    SymbolicScalar symValidHL1;   // 当前L1 tile H维真实数据量
    SymbolicScalar symPadLeftL1;  // 当前L1 tile左pad(W轴, 窗口坐标)
    SymbolicScalar symSrcOffsetW; // 当前L1 tile GM搬运W偏移(view坐标, >=0)
    SymbolicScalar symValidWL1;   // 当前L1 tile W维真实数据量
    // load3dv2 params
    int64_t repeatStride = 0;
    int64_t repeatTime = 1;
    int64_t wStride = 0;
};

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op);

} // namespace Conv

namespace ConvBp {
constexpr const int NCL_N_IDX = 0;
constexpr const int NCL_C_IDX = 1;
constexpr const int NCL_L_IDX = 2;
constexpr const int NCHW_N_IDX = 0;
constexpr const int NCHW_C_IDX = 1;
constexpr const int NCHW_H_IDX = 2;
constexpr const int NCHW_W_IDX = 3;
constexpr const int NCDHW_N_IDX = 0;
constexpr const int NCDHW_C_IDX = 1;
constexpr const int NCDHW_D_IDX = 2;
constexpr const int NCDHW_H_IDX = 3;
constexpr const int NCDHW_W_IDX = 4;
constexpr const int NDC1HWC0_N_IDX = 0;
constexpr const int NDC1HWC0_D_IDX = 1;
constexpr const int NDC1HWC0_C1_IDX = 2;
constexpr const int NDC1HWC0_H_IDX = 3;
constexpr const int NDC1HWC0_W_IDX = 4;
constexpr const int NDC1HWC0_C0_IDX = 5;
constexpr const int NC1HWC0_N_IDX = 0;
constexpr const int NC1HWC0_C1_IDX = 1;
constexpr const int NC1HWC0_H_IDX = 2;
constexpr const int NC1HWC0_W_IDX = 3;
constexpr const int NC1HWC0_C0_IDX = 4;
constexpr const int INPUT_GRAD_OUTPUT_IDX = 0;
constexpr const int INPUT_WEIGHT_IDX = 1;
constexpr const int INPUT_BIAS_IDX = 2;

const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string CONV_PADDINGS_ATTR = OP_ATTR_PREFIX + "paddings";
const std::string CONV_DILATIONS_ATTR = OP_ATTR_PREFIX + "dilations";
const std::string CONV_STRIDES_ATTR = OP_ATTR_PREFIX + "strides";
const std::string CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_gradOutput_shape";
const std::string CONV_ORI_WEIGHT_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_weight_shape";
const std::string CONV_ORI_RES_SHAPE_ATTR = OP_ATTR_PREFIX + "ori_res_shape";
const std::string CONV_BIAS_ATTR = OP_ATTR_PREFIX + "bias_flag";
const std::string CONV_3D_FLAG = OP_ATTR_PREFIX + "is_conv3d";
const std::vector<int64_t> CONV2D_ATTR_DEFAULT_LIST = {1, 1};
const std::vector<int64_t> CONV3D_ATTR_DEFAULT_LIST = {1, 1, 1};
const std::vector<int64_t> CONV2D_PAD_ATTR_DEFAULT_LIST = {0, 0, 0, 0};
const std::vector<int64_t> CONV3D_PAD_ATTR_DEFAULT_LIST = {0, 0, 0, 0, 0, 0};

class LoadStoreConvBpOpAttributeKey {
public:
    static const std::string copyInMode;
    static const std::string copyOutMode;
    static const std::string isGradOutput;
};

struct ConvBpAttrParam {
    std::vector<int64_t> paddings = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1, 1};
    std::vector<int64_t> dilations = {1, 1, 1};
    std::vector<int64_t> oriGradOutputShape = {0, 0, 0, 0};
    std::vector<int64_t> oriWeightShape = {0, 0, 0, 0};
    std::vector<int64_t> oriResShape = {0, 0, 0, 0};
    std::vector<SymbolicScalar> dynValidResShape;
    int64_t groups = 0;
    bool isConv1D = false;
    bool isConv3D = false;
    int32_t oriDim = 1;
    bool hasBias = false;
    bool isInOutTensorNZ = false;

    ConvBpAttrParam() = default;

    ConvBpAttrParam(std::vector<int64_t> paddingsList, std::vector<int64_t> stridesList,
                    std::vector<int64_t> dilationsList, int64_t groupsValue)
    {
        paddings = paddingsList;
        strides = stridesList;
        dilations = dilationsList;
        groups = groupsValue;
    }
};

struct ConvBpGraphNodes {
    LogicalTensorPtr gradOutputTensorPtr = nullptr;
    LogicalTensorPtr weightTensorPtr = nullptr;
    LogicalTensorPtr cL0PartialSumPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr resTensorPtr = nullptr;
};

struct ConvBpTileInfo {
    int64_t orgBatch = 0;
    int64_t orgCin = 0;
    int64_t orgCout = 0;
    int64_t orgDin = 0;
    int64_t orgDout = 0;
    int64_t orgHin = 0;
    int64_t orgWin = 0;
    int64_t orgHinWin = 0;
    int64_t orgHout = 0;
    int64_t orgWout = 0;
    int64_t orgHoutWout = 0;
    int64_t orgKh = 0;
    int64_t orgKw = 0;
    int64_t orgKd = 0;
    int64_t expandHout = 0;
    int64_t expandWout = 0;
    int64_t bpPadUp = 0;
    int64_t bpPadDown = 0;
    int64_t bpPadLeft = 0;
    int64_t bpPadRight = 0;
    int64_t kPerGroup = 0;
    int64_t cinPerGroup = 0;
    int64_t cin1PerGroup = 0;
    int64_t coutPerGroup = 0;
    int64_t cout1PerGroup = 0;
    int64_t cin0 = 0;
    int64_t cout0 = 0;

    int64_t mL1 = 0;
    int64_t kL1 = 0;
    int64_t nL1 = 0;
    int64_t mL0 = 0;
    int64_t kL0 = 0;
    int64_t nL0 = 0;
};

struct ConvBpIterInfo {
    int64_t groupOffset = 0;
    int64_t batchOffset = 0;
    int64_t dinOffset = 0;

    int64_t doutOffset = 0;
    int64_t dkOffset = 0;
    int64_t hinL1Offset = 0;
    int64_t winL1Offset = 0;
    int64_t houtL1Offset = 0;
    int64_t woutL1Offset = 0;
    int64_t mL1Offset = 0;
    int64_t nL1Offset = 0;
    int64_t kL1Offset = 0;

    int64_t houtL0Offset = 0;
    int64_t woutL0Offset = 0;
    int64_t mL0Offset = 0;
    int64_t nL0Offset = 0;
    int64_t kL0Offset = 0;

    int64_t hinL1Size = 0;
    int64_t winL1Size = 0;
    int64_t houtL1Size = 0;
    int64_t woutL1Size = 0;
    int64_t mL1Size = 0;
    int64_t nL1Size = 0;
    int64_t kL1Size = 0;
    int64_t mL0Size = 0;
    int64_t nL0Size = 0;
    int64_t kL0Size = 0;

    bool aL1UpadateFlag = false;
    bool bL1UpadateFlag = false;
    bool isFirstK = false;
    bool isLastK = false;
    // load3dv2 params
    int64_t repeatStride = 0;
    int64_t repeatTime = 1;
    int64_t wStride = 0;
};

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op);

} // namespace ConvBp
} // namespace npu::tile_fwk
