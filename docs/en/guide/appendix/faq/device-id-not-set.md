# Device ID Not Set for Operator Execution

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:14:52.882Z pushedAt=2026-10-27T02:41:48.384Z nl=2 -->

## Symptoms

When an operator is executed on the Ascend AI processor, the execution fails and the following error message is displayed:

```text
2025-12-17 14:31:32.491 E | fail get device id, check if set device id
2025-12-17 14:31:32.492 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | aclmdlRICaptureGetInfo failed, return[100000]
```

## Possible Causes

The user-defined operator is not decorated with `@jit`, or the device ID for the current operator execution is not explicitly set using the TorchNPU API.

## Solution

Set the device ID before executing the operator. For example:

```python
def test_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))  # Obtain the expected device ID from the environment variable.
    torch.npu.set_device(device_id)  # Explicitly set the device ID.
    ....
```
