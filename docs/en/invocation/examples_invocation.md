# Running Samples

## Running Environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

For more examples, see the sample code in the `examples/` directory.

## Viewing Results

After this basic sample runs successfully, compilation and running outputs are generated in the `${work_path}/output/` directory. The outputs include the [compute graph](../guide/appendix/glossary.md) and [swimlane diagram](../guide/appendix/glossary.md). You can view the compute graph and swimlane diagram in VS Code and associate them with the code by using the PyPTO Toolkit plugin. For details about how to use the Toolkit, refer to [Quick Start - Viewing the Compute Graph](../guide/quick_start/tensor/quick_start.md#viewing-compute-graphs) and [Quick Start - Viewing the Swimlane Diagram](../guide/quick_start/tensor/quick_start.md#viewing-the-swimlane-diagram)

## Quick Start

The following is a simple PyPTO usage example:

```python
import pypto
import torch
import torch_npu

shape = (1, 4, 1, 64)

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel(
    x: pypto.Tensor([...], pypto.DT_FP32),
    y: pypto.Tensor([...], pypto.DT_FP32),
    out: pypto.Tensor([...], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out[:] = x + y

if __name__ == "__main__":
    torch.npu.set_device(0)
    device = "npu:0"

    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)

    # Run the computation and view the result
    add_kernel(x, y, output)
    print(f"Output shape: {output.shape}")
```

- You can view the running result by directly checking the value of the output tensor

For the complete sample, see [hello_world.py](../../../examples/00_hello_world/hello_world.py).
