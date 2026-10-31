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
 * \file distributed.cpp
 * \brief
 */

#include "pybind_common.h"

#include "machine/runtime/distributed/window_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::Distributed;

namespace pypto {
void BindDistributed(py::module_& m)
{
    py::class_<ShmemTensor>(m, "ShmemTensor")
        .def(py::init<>())
        .def_readwrite("group", &ShmemTensor::group)
        .def_readwrite("worldSize", &ShmemTensor::worldSize)
        .def_readwrite("data", &ShmemTensor::data)
        .def_readwrite("signal", &ShmemTensor::signal);

    m.def(
        "CreateShmemTensor",
        [](const char* group, int64_t worldSize, DataType dataType, const Shape& shape, ShmemTensor& t) {
            return Distributed::CreateShmemTensor(group, worldSize, dataType, shape, t);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("dataType"), py::arg("shape"), py::arg("t"),
        "Create shmem data.");

    m.def(
        "CreateShmemSignal",
        [](const char* group, int64_t worldSize, ShmemTensor& t) {
            return Distributed::CreateShmemSignal(group, worldSize, t);
        },
        py::arg("group"), py::arg("worldSize"), py::arg("t"), "Create shmem signal data.");

    m.def(
        "ShmemView",
        [](const ShmemTensor& operand, const std::vector<int64_t>& shapes, const py::sequence& offsets) {
            bool has_symbolic = false;
            for (const auto& item : offsets) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_offsets;
                symbolic_offsets.reserve(py::len(offsets));
                for (const auto& item : offsets) {
                    symbolic_offsets.push_back(item.cast<SymbolicScalar>());
                }
                return Distributed::ShmemView(operand, shapes, symbolic_offsets);
            } else {
                std::vector<int64_t> int_offsets;
                int_offsets.reserve(py::len(offsets));
                for (const auto& item : offsets) {
                    int_offsets.push_back(item.cast<int64_t>());
                }
                return Distributed::ShmemView(operand, shapes, int_offsets);
            }
        },
        py::arg("operand"), py::arg("shapes"), py::arg("offsets"), "Create shmem view.");

    m.def(
        "ShmemView",
        [](const ShmemTensor& operand, const std::vector<int64_t>& shapes, const py::sequence& newValidShapes,
           const py::sequence& newOffsets) {
            std::vector<SymbolicScalar> symbolic_newValidShapes;
            symbolic_newValidShapes.reserve(py::len(newValidShapes));
            for (const auto& item : newValidShapes) {
                symbolic_newValidShapes.push_back(item.cast<SymbolicScalar>());
            }
            std::vector<SymbolicScalar> symbolic_offsets;
            symbolic_offsets.reserve(py::len(newOffsets));
            for (const auto& item : newOffsets) {
                symbolic_offsets.push_back(item.cast<SymbolicScalar>());
            }
            return Distributed::ShmemView(operand, shapes, symbolic_newValidShapes, symbolic_offsets);
        },
        py::arg("operand"), py::arg("shapes"), py::arg("newValidShapes"), py::arg("newOffsets"),
        "Create shmem view with valid shapes and offsets.");

    m.def(
        "ShmemPut",
        [](const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, Distributed::AtomicType putOp,
           const Tensor& pred) { return Distributed::ShmemPut(src, dst, dstRank, putOp, pred); },
        py::arg("src"), py::arg("dst"), py::arg("dstRank"), py::arg("putOp"), py::arg("pred"),
        "Put tensor to shmem with rank.");

    m.def(
        "ShmemGet",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
           DataType targetDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemGet(src, srcRank, pred, targetDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("targetDataType") = DataType::DT_BOTTOM,
        "Get shmem data with rank.");

    m.def(
        "ShmemSignal",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const SymbolicScalar& targetRank, int32_t signal,
           Distributed::AtomicType sigOp,
           const Tensor& pred) { return Distributed::ShmemSignal(src, srcRank, targetRank, signal, sigOp, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("targetRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal shmem with consumer rank.");

    m.def(
        "ShmemSignalAll",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, int32_t signal, Distributed::AtomicType sigOp,
           const Tensor& pred) { return Distributed::ShmemSignalAll(src, srcRank, signal, sigOp, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("signal"), py::arg("sigOp"), py::arg("pred"),
        "Signal all ranks in shmem.");

    m.def(
        "ShmemWaitUntil",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, OpType cmp, int32_t cmpValue, bool clearSignal,
           const Tensor& pred) { return Distributed::ShmemWaitUntil(src, srcRank, cmp, cmpValue, clearSignal, pred); },
        py::arg("src"), py::arg("srcRank"), py::arg("cmp"), py::arg("cmpValue"), py::arg("clearSignal"),
        py::arg("pred"), "Wait shmem signal.");

    m.def(
        "ShmemClearData", [](const ShmemTensor& src, Tensor& pred) { return Distributed::ShmemClearData(src, pred); },
        py::arg("src"), py::arg("pred"), "Clear shmem data.");

    m.def(
        "ShmemClearSignal",
        [](const ShmemTensor& src, Tensor& pred) { return Distributed::ShmemClearSignal(src, pred); }, py::arg("src"),
        py::arg("pred"), "Clear shmem signal.");

    m.def(
        "ShmemBarrier", [](const ShmemTensor& src, const Tensor& pred) { return Distributed::ShmemBarrier(src, pred); },
        py::arg("src"), py::arg("pred"), "Barrier on shmem.");

    m.def(
        "ShmemLoad",
        [](const ShmemTensor& src, const SymbolicScalar& srcRank, const Tensor& pred,
           DataType nonShmemDataType = DataType::DT_BOTTOM) {
            return Distributed::ShmemLoad(src, srcRank, pred, nonShmemDataType);
        },
        py::arg("src"), py::arg("srcRank"), py::arg("pred"), py::arg("nonShmemDataType") = DataType::DT_BOTTOM,
        "Load shmem data to local.");

    m.def(
        "ShmemStore",
        [](const Tensor& src, const ShmemTensor& dst, const SymbolicScalar& dstRank, Distributed::AtomicType putOp,
           const Tensor& pred) { return Distributed::ShmemStore(src, dst, dstRank, putOp, pred); },
        py::arg("src"), py::arg("dst"), py::arg("dstRank"), py::arg("putOp"), py::arg("pred"),
        "Store local tensor to shmem.");

    m.def(
        "GetSymbolicScalarPeId", [](std::string group) { return GetHcclRankId(group); }, py::arg("group"),
        "Get local rank id by groupname.");

    py::module_ dist = m.def_submodule("dist");
#ifdef BUILD_WITH_CANN // mirrors the window_context.cpp guard; keeps no-CANN builds linking
    py::class_<WindowContext, std::shared_ptr<WindowContext>>(dist, "Window", py::dynamic_attr())
        .def("rank", &WindowContext::Rank, "This rank's index in the group.")
        .def("nrank", &WindowContext::NRank, "Group size.")
        .def("size", &WindowContext::Size, "Payload size in bytes (data region only).")
        .def("num_signals", &WindowContext::NumSignals, "Reserved signal-slot count.")
        .def("data_ptr", &WindowContext::DataPtr,
             "Device address of the device-side Window descriptor (host handle; in-kernel "
             "win.data_ptr() reads the payload base instead).")
        .def("support_multi_issuer", &WindowContext::SupportsMultiIssuer,
             "True when every peer shares one local UBC_CTP endpoint, so any AIV may issue engine "
             "writes to any peer; False when each peer needs a single issuer AIV.")
        .def("__str__", &WindowContext::ToString)
        .def(
            "close",
            [](WindowContext& win) {
                // Collective teardown blocks on peers; don't stall other Python threads.
                py::gil_scoped_release gil;
                win.Close();
            },
            "Release resources; all-rank collective.");

    dist.def(
        "create_window",
        [](int32_t rank, int32_t nrank, int64_t size, int64_t bufferSize, int64_t numSignals, int64_t deviceAddr,
           int64_t streamAddr, const std::string& hcclGroupName, const std::string& name) {
            std::shared_ptr<WindowContext> win;
            {
                // Collective setup blocks for seconds (AllGather barriers, channel connect);
                // don't stall other Python threads while polling.
                py::gil_scoped_release gil;
                win = CreateWindow(rank, nrank, size, bufferSize, numSignals, deviceAddr, streamAddr, hcclGroupName,
                                   name);
            }
            return win;
        },
        py::arg("rank"), py::arg("nrank"), py::arg("size"), py::arg("buffer_size"), py::arg("num_signals"),
        py::arg("device_addr"), py::arg("stream"), py::arg("hccl_name"), py::arg("name") = std::string(),
        "Register a symmetric window on the HCCL comm and build its connections.");

    dist.def(
        "dump_topology",
        [](const std::string& hcclGroupName, int32_t rank, int32_t nrank, int32_t layer) {
            return GetGroupTopology(hcclGroupName, rank, nrank, layer);
        },
        py::arg("hccl_name"), py::arg("rank"), py::arg("nrank"), py::arg("layer") = -1,
        "Dump the rank-graph topology of an HCCL comm: per net layer its interconnect topology, the ranks "
        "reachable from this rank, and the links to each peer. layer < 0 dumps every layer.");
#endif
}

} // namespace pypto
