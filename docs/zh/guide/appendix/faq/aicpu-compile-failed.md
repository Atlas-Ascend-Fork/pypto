# AI CPU Kernel编译失败

## 问题现象描述

抛出RuntimeFailure，报错信息包含`AICPU kernel compile failed`、`AICPU host compile failed`或`AICPU link failed`，并附带完整的bisheng命令与stderr。

编译步骤失败（附带完整命令与编译器输出）：

```text
pypto_pro._errors.RuntimeFailure: AICPU kernel compile failed for kernel 'batch_row_plan' (rc=1)
command: /usr/local/Ascend/ascend-toolkit/latest/bin/bisheng -O2 -c -std=c++17 ... -x aicpu .../batch_row_plan_aicpu.cpp
stdout:
stderr:
.../batch_row_plan_aicpu.cpp:xx:xx: error: ...
```

环境前置条件不满足（在编译命令执行前抛出）：

```text
pypto_pro._errors.RuntimeFailure: AICPU build needs ASCEND_HOME_PATH with a CANN install (no <prefix>-linux directory found)
```

```text
pypto_pro._errors.RuntimeFailure: AICPU build needs the hcc toolchain under '${ASCEND_HOME_PATH}/toolkit/toolchain/hcc' (not found)
```

## 可能原因

- `ASCEND_HOME_PATH`未设置或指向的CANN安装不完整（缺少`<prefix>-linux`目录、`toolkit/toolchain/hcc`工具链）。
- bisheng编译器缺失或版本不匹配。
- Kernel函数体使用了AI CPU工具链不接受的构造（报错stderr中可见具体编译错误）。

## 处理步骤

1. 确认环境变量：`ASCEND_HOME_PATH`指向CANN toolkit安装根目录（含bin/bisheng）。
2. 核对报错中附带的bisheng命令与stderr，定位是环境问题还是源码问题。
3. 检查产物目录中的`{op}_aicpu.cpp`与`{op}_aicpu_host.cpp`生成源码，确认失败位置，产物说明参见[JIT编译](../../programming_guide/pro/development/compilation_and_execution/JIT_compilation.md#编译产物)。
