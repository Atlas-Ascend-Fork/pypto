# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE _MachineInterface_Srcs *.cpp)
# InitLogSwitch 依赖 HardBranchManager, 单独编进 interface so, 避免运行时依赖 backend server so
list(APPEND _MachineInterface_Srcs
    ${PTO_FWK_SRC_ROOT}/framework/src/machine/device/dynamic/aicpu_instrumentation.cpp
)

add_library(tile_fwk_server_interface SHARED)
target_sources(tile_fwk_server_interface PRIVATE ${_MachineInterface_Srcs})
target_compile_options(tile_fwk_server_interface
        PRIVATE
            -Wall
            -Wextra
            -std=gnu++17
            -rdynamic
            -O3
            -march=armv8-a
            -shared
            -fPIC
            -fvisibility=hidden
            -D__DEVICE__
            -Werror
            -fno-common
            -fstack-protector-strong
)
target_include_directories(tile_fwk_server_interface
        PRIVATE
            ${PTO_FWK_SRC_ROOT}/framework/include
            ${PTO_FWK_SRC_ROOT}/framework/src
            ${PTO_FWK_SRC_ROOT}/framework/src/interface
            ${PTO_FWK_SRC_ROOT}/framework/src/interface/machine/device
)
target_link_libraries(tile_fwk_server_interface
        PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:ascend_hal_headers>
            $<BUILD_INTERFACE:atrace_headers>
            $<BUILD_INTERFACE:slog_headers>
            -Wl,--no-as-needed
            c_sec
            slog_stub
            -ldl
)
target_link_options(tile_fwk_server_interface
        PRIVATE
            -Wl,-z,relro
            -Wl,-z,now
            -s
)
set_target_properties(tile_fwk_server_interface PROPERTIES OUTPUT_NAME pypto_aicpu)

if (ENABLE_SIGN)
    # 仅签名(CANN 工程)构建产出并打包 aicpu kernel tar; 开源构建不带该 tar,
    # 安装时无新 tar 可装, 自然保留已安装的旧 tar。
    if (NOT DEFINED CANN_VERSION_pypto_VERSION)
        include(${PTO_FWK_SRC_ROOT}/version.cmake)
    endif ()

    set(PYPTO_AICPU_TAR_GZ_NAME "pypto_aicpu.tar.gz")
    set(PYPTO_AICPU_TAR_GZ "${CMAKE_CURRENT_BINARY_DIR}/${PYPTO_AICPU_TAR_GZ_NAME}")
    add_custom_command(
        OUTPUT ${PYPTO_AICPU_TAR_GZ}
        COMMAND echo "Begin to compress ${PYPTO_AICPU_TAR_GZ_NAME} from $<TARGET_FILE:tile_fwk_server_interface>"
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tile_fwk_server_interface> ${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device
        COMMAND tar czf ${PYPTO_AICPU_TAR_GZ_NAME} -C ${CMAKE_CURRENT_BINARY_DIR} aicpu_kernels_device
        COMMAND echo "${PYPTO_AICPU_TAR_GZ} has been generated."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS tile_fwk_server_interface
        COMMENT "Packing ${PYPTO_AICPU_TAR_GZ_NAME}"
        VERBATIM
    )
    add_custom_target(pack_pypto_aicpu ALL DEPENDS ${PYPTO_AICPU_TAR_GZ})

    add_cann_sign_file(
        INPUT "${PYPTO_AICPU_TAR_GZ}"
        CONFIG "${PTO_FWK_SRC_ROOT}/scripts/package/pypto/sign.xml"
        RESULT_VAR "pypto_aicpu_sign_file"
        VERSION "${CANN_VERSION_pypto_VERSION}"
        DEPENDS pack_pypto_aicpu
    )
    install(FILES ${pypto_aicpu_sign_file} DESTINATION lib COMPONENT pypto OPTIONAL)
endif ()

install(FILES ${PTO_FWK_SRC_ROOT}/framework/src/machine/runtime/runner/pypto_aicpu.json DESTINATION lib COMPONENT pypto)
