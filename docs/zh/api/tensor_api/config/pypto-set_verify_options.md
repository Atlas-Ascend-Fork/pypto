# pypto.set\_verify\_options

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="9030" id4 -->
- Kirin9030处理器系列产品：支持
<!-- end id4 -->
<!-- npu="x90" id5 -->
- KirinX90处理器系列产品：支持
<!-- end id5 -->

## 功能说明

设置精度调试Verify特性的自检开关、及特性对应的功能选项。

## 函数原型

```python
set_verify_options(*,
                   enable_pass_verify: Optional[bool] = None,
                   pass_verify_save_tensor: Optional[bool] = None,
                   pass_verify_save_tensor_dir: Optional[str] = None,
                   pass_verify_pass_filter: Optional[List[str]] = None,
                   pass_verify_error_tol: Optional[List[float]] = None,
                   ) -> None
```

## 参数说明

| 参数名                          | 输入/输出 | 说明                                                                 |
|---------------------------------|-----------|----------------------------------------------------------------------|
| enable_pass_verify              | 输入      | 含义：总开关，决定所有pass_verify_*选项、接口是否有效。 <br> 说明：True代表开启。 <br> 类型：bool <br> 取值范围：True/False <br> 默认值：False |
| pass_verify_save_tensor         | 输入      | 含义：配置是否将模拟计算的逐算子中间数据额外存盘。 <br> 说明：True代表存盘；校验结果目录及结果文件的生成不受该开关影响。 <br> 类型：bool <br> 取值范围：True/False <br> 默认值：False |
| pass_verify_save_tensor_dir     | 输入      | 含义：配置检测结果及数据的保存路径，结果文件生成在该路径下的verify子目录中。 <br> 说明：设定绝对路径的字符串。 <br> 类型：str <br> 默认值：<br> {RUNNING_DIR}/output/output_{TS} |
| pass_verify_pass_filter         | 输入      | 含义：配置待自检的Pass名称列表。 <br> 说明：合法的Pass名称。 <br> 不指定则默认校验pass：["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"]；指定"all"则校验所有pass，指定[]不校验pass只校验tensor_graph；指定非法名称则忽略。 <br> 类型：List[str] <br> 默认值：<br> ["ExpandFunction", "ProcessAtomic", "L1CopyInReuseMerge", "InferDynShape", "PreGraphProcess", "InferParamIndex", "CodegenPreproc"] |
| pass_verify_error_tol           | 输入      | 含义：配置精度工具对比精度需要用到的rtol和atol。 <br> 说明：List中的第一个值是rtol，第二个值为atol；List长度不等于2时，使用默认值。 <br> 类型：List[float] <br> 默认值：[1e-3, 1e-3] |

## 返回值说明

void：Set方法无返回值。设置成功后在后续算子编译时生效。

## 约束说明

- 需在算子编译前设置，对随后编译的算子生效。
- 开启后校验在算子编译期间于主机侧模拟执行，不改变算子的编译与运行结果。
- 开启时（enable_pass_verify=True）会自动加载模拟计算库，无需用户手动处理：JIT调用模式下由框架在算子编译时加载；非JIT场景（如直接录制pypto.function、IR编译流程）在本接口调用时加载。首次加载会在当前进程内在线编译（依赖torch环境），耗时较长。
- 校验结果不以异常形式上报：无论PASS/FAIL均不中断算子编译；需在pass_verify_save_tensor_dir（默认{RUNNING_DIR}/output/output_{TS}）目录下的verify子目录中查看interpreter.log、verify_graph_result_brief.csv等结果文件，其中ErrCode: FB4001表示校验结果不一致。

## 调用示例

```python
verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
        "pass_verify_save_tensor_dir": "/LARGE/DRIVE/DIR",
        }
pypto.set_verify_options(**verify_options)
```
