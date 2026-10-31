# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any


class Window:
    """Symmetric window registered on an HCCL comm (host-side handle)."""

    # torch uint8 tensor backing the window; set by pld.create_window.
    device_buffer: Any

    def rank(self) -> int: ...
    def nrank(self) -> int: ...
    def size(self) -> int: ...
    def num_signals(self) -> int: ...
    def data_ptr(self) -> int: ...
    def support_multi_issuer(self) -> bool: ...
    def close(self) -> None: ...
    def __str__(self) -> str: ...


def create_window(
    rank: int,
    nrank: int,
    size: int,
    buffer_size: int,
    num_signals: int,
    device_addr: int,
    stream: int,
    hccl_name: str,
    name: str = ...,
) -> Window: ...


def dump_topology(hccl_name: str, rank: int, nrank: int, layer: int = ...) -> str: ...
