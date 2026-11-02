# CANN Package Incompatibility

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:14:05.076Z pushedAt=2026-10-27T02:41:41.518Z nl=2 -->

## Symptoms

The following error message appears during operator on-device execution:

```text
ErrorTracking callback in, task_id = 0, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 0, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
ErrorTracking callback in, task_id = 1, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 1, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
```

And the device log shows an error similar to the following, indicating that the function pointer is null:

```text
~/ascend/log/debug/device-0/device-6495_20251222194004973.log
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.899.541 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5680][AICPU_PROCESSER] Get KFC DynTileFwkKernelServerInit api success, but func is nullptr: (null)
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.902.745 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5681][AICPU_PROCESSER] Get KFC DynTileFwkKernelServer api success, but func is nullptr: (null)
```

## Possible Causes

The PyPTO driver package supports versions later than 25.2.0, and the CANN package supports versions later than 8.5.0.

## Solution

Check the version information in the driver package installation directory, for example:

```text
/usr/local/Ascend/driver/version.info
    Version=25.3.rc1
    ascendhal_version=7.35.23
    aicpu_version=1.0
    tdt_version=1.0
    log_version=1.0
    prof_version=2.0
    dvppkernels_version=1.1
    tsfw_version=1.0
    Innerversion=V100R001C23SPC002B212
    compatible_version=[V100R001C19],[V100R001C20],[V100R001C21],[V100R001C22],[V100R001C23]
    compatible_version_fw=[7.0.0,8.9.9]
    package_version=25.3.rc1
```

Similarly, you can check the version information in the **opp** package under the CANN package installation directory, for example:

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    Version=8.5.0.2.220
    version_dir=8.5.0
    timestamp=20251117_000024591
    required_package_amct_acl_version="8.5"
```

Check whether the driver package and CANN package meet the version requirements using the method described above. If they do not, upgrade the corresponding versions.
