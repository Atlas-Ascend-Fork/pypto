# Running Samples

## Simulation Environment (Without Real NPU Hardware)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd examples/00_hello_world
python3 hello_world.py --run_mode=sim
```

## Real Running Environment (With Real NPU Hardware)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
cd examples/00_hello_world
python3 hello_world.py --run_mode=npu
```

For more examples, see the sample code in the `examples/` directory.

## Viewing Results

After this basic sample runs successfully, compilation and running outputs are generated in the `${work_path}/output/` directory. The outputs include the [compute graph](../../zh/guide/appendix/glossary.md) and [swimlane diagram](../../zh/guide/appendix/glossary.md). You can view the compute graph and swimlane diagram in VS Code and associate them with the code by using the PyPTO Toolkit plugin. For details about how to use the Toolkit, refer to [Quick Start - Viewing the Compute Graph](../../zh/guide/quick_start/tensor/quick_start.md) and [Quick Start - Viewing the Swimlane Diagram](../../zh/guide/quick_start/tensor/quick_start.md)

## Quick Start

The following is a simple PyPTO usage example. You can use the `--run_mode` parameter to specify whether to run the simulation example or the real environment example:

```python
import pypto
import torch
import argparse

shape = (1, 4, 1, 64)

# Create the compute kernel based on the run mode
def create_add_kernel(run_mode: str):
    mode = pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        x: pypto.Tensor([...], pypto.DT_FP32),
        y: pypto.Tensor([...], pypto.DT_FP32),
        out: pypto.Tensor([...], pypto.DT_FP32),
    ):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = x + y

    return add_kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu", "sim"])
    args = parser.parse_args()

    # Prepare the input data
    device = "cpu"
    if args.run_mode == "npu":
        import torch_npu
        torch.npu.set_device(0)
        device = "npu:0"

    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)
    output = torch.empty(shape, dtype=torch.float32, device=device)

    # Run the computation and view the result
    create_add_kernel(args.run_mode)(x, y, output)
    print(f"Output shape: {output.shape}")
```

- For the real environment or precision simulation, you can directly view the running result by checking the value of the output tensor
- For performance simulation, view the simulation result through the swimlane diagram in the `output/` directory

For the complete sample, see [hello_world.py](../../../examples/00_hello_world/hello_world.py).
