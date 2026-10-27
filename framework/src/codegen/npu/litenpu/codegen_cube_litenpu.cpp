/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_cube_litenpu.cpp
 * \brief kirin9050 matmul 宏指令 codegen（与通用 matmul codegen 解耦，仅 litenpu 使用）
 */

#include "codegen_op_litenpu.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {

namespace {
bool IsMacroInstrMatmulOpcode(Opcode opCode)
{
    return opCode == Opcode::OP_A_MUL_B_EPILOGUE || opCode == Opcode::OP_A_MULACC_B_EPILOGUE ||
           opCode == Opcode::OP_A_MUL_BT_EPILOGUE || opCode == Opcode::OP_A_MULACC_BT_EPILOGUE ||
           opCode == Opcode::OP_AT_MUL_B_EPILOGUE || opCode == Opcode::OP_AT_MUL_BT_EPILOGUE;
}
} // namespace

std::string CodeGenOpLiteNPU::PrintMatmulTileTensor(bool zeroC) const
{
    if (!IsMacroInstrMatmulOpcode(opCode)) {
        return CodeGenOpNPU::PrintMatmulTileTensor(zeroC);
    }

    // C/A/B 按操作数位置取参名（0=C, 1=A, 2=B）：kirin9050 无 L0A/L0C buffer，A/C 驻留 L1，
    // 不能按 BUF_L0C/BUF_L0A 内存类型查找。
    const std::string cName = QueryTileTensorNameByIdx(0);
    const std::string aName = QueryTileTensorNameByIdx(1);
    const std::string bName = QueryTileTensorNameByIdx(2);
    std::vector<std::string> paramList = {cName, aName, bName};

    bool isGemv = false;
    GetOpAttr(OpAttributeKey::isGemv, isGemv);
    bool isConv = false;
    GetOpAttr(OpAttributeKey::isConv, isConv);
    int64_t transModeNum = 0;
    GetOpAttr(OpAttributeKey::transMode, transModeNum);
    std::string transModeStr = "TransMode::CAST_NONE";
    if (transModeNum == static_cast<int64_t>(TransMode::CAST_RINT)) {
        transModeStr = "TransMode::CAST_RINT";
    } else if (transModeNum == static_cast<int64_t>(TransMode::CAST_ROUND)) {
        transModeStr = "TransMode::CAST_ROUND";
    }
    const bool kAlignFlag = !isConv;

    std::unordered_map<OperandType, std::string> tensorWithMemType;
    for (int i = 0; i < operandCnt; i++) {
        tensorWithMemType.emplace(operandType[i], QueryTileTensorNameByIdx(i));
    }
    const bool hasBias = tensorWithMemType.count(OperandType::BUF_BT) > 0;

    // kirin9050 宏指令 pre-stage quant：scale_tensor（vector_quant_flag）复用为 quant_pre，
    // quant_pre 恒为最后一个操作数，走 TMatmulQuant/TGemvQuant 精简接口（tileOpName 已由
    // UpdateGemvTileOpName 设置）。
    bool hasQuantPre = false;
    GetOpAttr(OP_ATTR_PREFIX + "vector_quant_flag", hasQuantPre);
    if (hasQuantPre) {
        bool pushedBiasOrPartial = false;
        if (operandCnt >= SHAPE_DIM4 && (hasBias || !zeroC)) {
            paramList.emplace_back(QueryTileTensorNameByIdx(3)); // bias（首K）或 partial sum（inplace 累加K）
            pushedBiasOrPartial = true;
        }
        if (operandCnt >= SHAPE_DIM5) {
            paramList.emplace_back(QueryTileTensorNameByIdx(4)); // quant_pre
        } else if (operandCnt >= SHAPE_DIM4 && !pushedBiasOrPartial) {
            paramList.emplace_back(QueryTileTensorNameByIdx(3)); // 首K无bias：idx3 即 quant_pre
        }
        std::ostringstream oss;
        oss << tileOpName;
        oss << WrapParamByAngleBrackets({std::to_string(zeroC)});
        oss << WrapParamByParentheses(paramList);
        oss << ";\n";
        return oss.str();
    }

    // 多 K 片末片：partial sum 是 int32 累加器、输出是 out dtype(fp16)，需要把 partial 操作数一并下发
    // （读 int32 acc 转 fp16）；中间片 inplace 累加仍是 3 参形式。
    const bool isLastKConvert = !zeroC && operandCnt >= SHAPE_DIM4 && operandDtype[0] != operandDtype[3];
    if (isLastKConvert) {
        paramList.emplace_back(QueryTileTensorNameByIdx(3));
        std::ostringstream oss;
        // 末片带 per-channel quant_pre(scale) 时走向量档反量化：scale 操作数（第 5 个）随末片下发，
        // 地址由 tileop 写入 m_fpc。s32 输出无反量化档，scale 不参与（与单 K 片路径一致）。
        const bool hasQuantPreOnLastK = operandCnt >= SHAPE_DIM5 && operandDtype[0] != DataType::DT_INT32;
        if (hasQuantPreOnLastK) {
            paramList.emplace_back(QueryTileTensorNameByIdx(4));
            oss << (isGemv ? "TGemvMacroAccFromQuant" : "TMatmulMacroAccFromQuant");
        } else {
            oss << (isGemv ? "TGemvMacroAccFrom" : "TMatmulMacroAccFrom");
        }
        oss << WrapParamByParentheses(paramList);
        oss << ";\n";
        return oss.str();
    }

    // 多 K 片 int32 累加器路径：K 中间片 partial sum 为 int32，走宏指令 acc 表单
    // （通用 TMatmul 不能实例化 int32 累加器）。
    const bool isInt32Acc = operandDtype[0] == DataType::DT_INT32;
    if (isInt32Acc) {
        // int8 GEMV 的首片/单片不能走宏 acc（gemvCtrl + M=1 只写前 16 列），保持 legacy TGEMV。
        const bool isInt8A = operandDtype[1] == DataType::DT_INT8;
        if (isGemv && zeroC && isInt8A) {
            return PrintMacroGenericMatmul(zeroC, paramList, transModeStr, kAlignFlag);
        }
        std::ostringstream oss;
        if (hasBias && zeroC) {
            // 首片带 bias：C(int32 acc) = A*B + bias（bias 经 BT 下发）
            paramList.emplace_back(QueryTileTensorNameByIdx(3));
            oss << (isGemv ? "TGemvMacroAccBias" : "TMatmulMacroAccBias");
            oss << WrapParamByParentheses(paramList);
        } else {
            // 首片零启动 / 中间片 inplace 累加
            oss << (isGemv ? "TGemvMacroAcc" : "TMatmulMacroAcc");
            oss << WrapParamByAngleBrackets({std::to_string(zeroC)});
            oss << WrapParamByParentheses(paramList);
        }
        oss << ";\n";
        return oss.str();
    }

    // C 为 out dtype（fp16 等）：单 K 片走通用 TMatmul/TGEMV（pto-isa DeqF16Scalar 等路径）
    return PrintMacroGenericMatmul(zeroC, paramList, transModeStr, kAlignFlag);
}

std::string CodeGenOpLiteNPU::PrintMacroGenericMatmul(bool zeroC, std::vector<std::string>& paramList,
                                                      const std::string& transModeStr, bool kAlignFlag) const
{
    std::ostringstream oss;
    bool hasBias = false;
    for (int i = 0; i < operandCnt; i++) {
        if (operandType[i] == OperandType::BUF_BT) {
            hasBias = true;
            paramList.emplace_back(QueryTileTensorNameByIdx(i));
            break;
        }
    }
    oss << tileOpName;
    if (hasBias) {
        oss << WrapParamByAngleBrackets({transModeStr, std::to_string(kAlignFlag)});
    } else {
        oss << WrapParamByAngleBrackets({std::to_string(zeroC), transModeStr, std::to_string(kAlignFlag)});
    }
    oss << WrapParamByParentheses(paramList);
    oss << ";\n";
    return oss.str();
}

} // namespace npu::tile_fwk
