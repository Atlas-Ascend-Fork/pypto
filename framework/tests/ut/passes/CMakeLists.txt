# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${PTO_Fwk_UTestNamePrefix}_passes)

file(GLOB_RECURSE _Sources src/*.cpp)

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        LINK_LIBRARIES
        LD_LIBRARIES_EXT pgo_vf
)
get_target_property(_pgo_vf_compile_definitions pgo_vf COMPILE_DEFINITIONS)
if("PGO_VF_SKIP" IN_LIST _pgo_vf_compile_definitions)
    set_source_files_properties(src/test_pgo_vf.cpp PROPERTIES COMPILE_DEFINITIONS PGO_VF_SKIP)
endif()
add_custom_command(
        TARGET ${_Target} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/pass/json
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/ut_json/*.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/pass/json
        COMMENT "Copy ${_Target} dependence config file to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config/pass/json"
)
