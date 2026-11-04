#include "pypto_launch.h"
#include "kernel.cpp"
extern "C" int64_t call_kernel(uint32_t blockDim, void* stream)
{
    const int64_t launchDim = pypto::ResolveLaunchBlockDim<1, 1>(blockDim, stream);
    if (launchDim < 0) {
        return launchDim;
    }
    blockDim = static_cast<uint32_t>(launchDim);
    probe<<<blockDim, nullptr, stream>>>();
    return launchDim;
}
