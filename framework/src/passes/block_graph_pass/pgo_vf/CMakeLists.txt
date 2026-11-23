# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_pgo_vf_impl_dir "${CMAKE_CURRENT_SOURCE_DIR}/pgo_vf_impl")

if(NOT EXISTS "${_pgo_vf_impl_dir}/CMakeLists.txt")
    add_library(pypto_pass_pgo_vf SHARED pgo_vf_api.cpp)
    target_compile_definitions(pypto_pass_pgo_vf PRIVATE PGO_VF_SKIP)
    target_compile_options(pypto_pass_pgo_vf PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
else()
    file(GLOB_RECURSE _pgo_vf_sources CONFIGURE_DEPENDS "${_pgo_vf_impl_dir}/src/*.cpp")
    add_library(pypto_pass_pgo_vf SHARED pgo_vf_api.cpp ${_pgo_vf_sources})
    target_include_directories(pypto_pass_pgo_vf PRIVATE ${_pgo_vf_impl_dir}/src)
    add_subdirectory(${_pgo_vf_impl_dir})
endif()

target_include_directories(pypto_pass_pgo_vf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PTO_FWK_SRC_ROOT}/framework/include
    ${PTO_FWK_SRC_ROOT}/framework/include/pypto_pro
    ${PTO_FWK_SRC_ROOT}/framework/src
    ${PTO_FWK_SRC_ROOT}/framework/src/interface
    ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device)
target_link_libraries(pypto_pass_pgo_vf PRIVATE tile_fwk_interface tile_fwk_utils tile_fwk_intf_pub)
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ENABLE_ASAN))
    target_link_options(pypto_pass_pgo_vf PRIVATE -Wl,--no-undefined -Wl,-z,defs)
endif()
set_target_properties(pypto_pass_pgo_vf PROPERTIES
    SOVERSION 1)
