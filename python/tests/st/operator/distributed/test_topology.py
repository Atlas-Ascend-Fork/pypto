#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and/or it is licensed under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Distributed tests for pld.dump_topology.

Main Functions:
    - topology_worker: dumps the topology of the default group (all layers, then layer 0)
    - test_dump_topology: spawns one process per rank and collects failures
"""

import multiprocessing as mp
import traceback

from distributed_config import DistributedConfig, collect_process_errors
import pypto_pro.language.distribute as pld
import pytest
import torch.distributed as dist


def topology_worker(worker_params, error_queue: mp.Queue):
    """
    Args:
        worker_params: list [config, logical_rank_id]
        error_queue: error queue
    """
    logical_rank_id = -1
    try:
        config, logical_rank_id = worker_params
        config.init_hccl_comm(logical_rank_id)
        group = dist.distributed_c10d._get_default_group()

        report = pld.dump_topology(group)
        assert isinstance(report, str) and "HCCL Topology" in report, f"bad report header: {report!r}"
        assert "Network Layer:" in report and "Topology     :" in report, f"no layer/topo info: {report!r}"
        assert f"Local Rank   : {logical_rank_id} / {config.world_size}" in report, f"bad rank line: {report!r}"
        assert "Transport" in report and "RemoteAddress" in report, f"no link table: {report!r}"

        # All layers must cover at least what the layer-0 dump shows.
        layer0 = pld.dump_topology(group, layer=0)
        assert "Network Layer: 0" in layer0, f"layer 0 missing: {layer0!r}"
        assert len(report) >= len(layer0), "all-layers dump shorter than the layer-0 dump"

        print(f"[rank {logical_rank_id}] dump_topology ok", flush=True)
    except Exception as e:
        if error_queue is not None:
            error_queue.put((logical_rank_id, str(e), traceback.format_exc()))
        raise


@pytest.mark.skip(reason="distributed python case")
@pytest.mark.world_size(4)
def test_dump_topology():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=4)

    error_queue = mp.Queue()
    processes = []
    for i in range(config.world_size):
        worker_params = [config, i]
        p = mp.Process(target=topology_worker, args=(worker_params, error_queue))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    collect_process_errors(processes, error_queue)


def main():
    test_dump_topology()


if __name__ == '__main__':
    main()
