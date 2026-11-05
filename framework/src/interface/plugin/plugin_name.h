/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file plugin_name.h
 * \brief Shared symbol names of the pluggable contract.
 */

#ifndef INTERFACE_PLUGIN_PLUGIN_NAME_H_
#define INTERFACE_PLUGIN_PLUGIN_NAME_H_

#define PLUGIN_GET_VERSION_NAME GetVersion

#define PLUGIN_STR_IMPL(x) #x
#define PLUGIN_STR(x) PLUGIN_STR_IMPL(x)

#endif // INTERFACE_PLUGIN_PLUGIN_NAME_H_
