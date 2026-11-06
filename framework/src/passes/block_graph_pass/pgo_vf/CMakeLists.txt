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
set(_pgo_vf_required_dirs
    pass_driver
    pass_interface
    reg_pass
    tile_pass)

set(_pgo_vf_skip FALSE)
foreach(_dir IN LISTS _pgo_vf_required_dirs)
    if(NOT IS_DIRECTORY "${_pgo_vf_impl_dir}/${_dir}")
        message(STATUS "pgo_vf: missing directory pgo_vf_impl/${_dir}, enabling PGO_VF_SKIP")
        set(_pgo_vf_skip TRUE)
        break()
    endif()
endforeach()

file(GLOB_RECURSE _sources CONFIGURE_DEPENDS *.cpp)
if(_pgo_vf_skip)
    list(FILTER _sources EXCLUDE REGEX "/pgo_vf_impl/")
endif()

if(NOT _sources)
    message(WARNING "pgo_vf: no sources under ${CMAKE_CURRENT_SOURCE_DIR}")
    return()
endif()

add_library(pgo_vf SHARED ${_sources})
if(_pgo_vf_skip)
    target_compile_definitions(pgo_vf PRIVATE PGO_VF_SKIP)
endif()
target_include_directories(pgo_vf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PTO_FWK_SRC_ROOT}/framework/include
    ${PTO_FWK_SRC_ROOT}/framework/src
    ${PTO_FWK_SRC_ROOT}/framework/src/interface
    ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device)
target_link_libraries(pgo_vf PRIVATE tile_fwk_interface tile_fwk_utils)
target_compile_options(pgo_vf PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_options(pgo_vf PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pgo_vf.map
    -Wl,--no-undefined
    -Wl,-z,defs)
set_target_properties(pgo_vf PROPERTIES
    OUTPUT_NAME pypto_pass_pgo_vf
    SOVERSION 1)
