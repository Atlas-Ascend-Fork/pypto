# F9XXXX

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:32:09.764Z pushedAt=2026-10-27T04:02:43.017Z nl=2 -->

## F91001 INVALID_CONFIG

**Error Description**

The configuration parameters passed in do not meet the simulator requirements, including the configuration string format not matching the `key=value` specification, the configuration name missing the `.` hierarchy delimiter, or the number of AI CPU cores not equal to the total AIC and AIV count.

**Possible Causes**

+ The configuration string does not match `key=value`.

   ```json
   # Error example - configuration string missing =value
   Model.deviceArch=
   ```

+ The `.` delimiter is missing between the configuration prefix and the subkey (for example, `Pipeline` should be `Pipeline.xxx`).

   ```json
   # Error example - configuration name missing . delimiter
   deviceArch="xxx"
   ```

+ `coreMachineNumberPerAICPU` is not equal to `aicNum + aivNum`.

   ```json
   # Error example - core count configuration mismatch
   Model.coreMachineNumberPerAICPU=12
   Model.cubeMachineNumberPerAICPU=4
   Model.vecMachineNumberPerAICPU=4
   ```

**Solution**

1. Ensure that the configuration string is in the `key=value` format, with key and value connected by `=`, for example, `Pipeline.enable=1`.

   ```json
   # Correct example - configuration string in key=value format
   Model.deviceArch="xxx"
   ```

2. Check that the configuration name contains the correct hierarchy prefix and `.` delimiter. For details, see the configuration definitions in each `xxx.h` file under `framework/src/cost_model/simulation/config/`.

3. Ensure that `coreMachineNumberPerAICPU` equals `aicNum + aivNum`, and recheck the hardware configuration parameters.

   ```json
   # Correct example - core count configuration correct
   Model.coreMachineNumberPerAICPU=12
   Model.cubeMachineNumberPerAICPU=4
   Model.vecMachineNumberPerAICPU=8
   ```

## F91002 INVALID_CONFIG_NAME

**Error Description**

The configuration name format is correct (containing the `.` delimiter), but the corresponding processor cannot be found in the simulation configuration dispatcher. That is, the configuration name is not in the registered list.

**Possible Causes**

+ The configuration name is misspelled.

   ```json
   # Error example - configuration name misspelled
   Model.devicearch="xxx"
   ```

+ A configuration name that does not exist or has not been registered in the dispatcher is used.

   ```json
   # Error example - an unregistered configuration name is used
   Model.notExist="xxx"
   ```

**Solution**

1. Check whether the configuration name is spelled correctly.
2. Refer to the list of valid configuration items registered by the dispatcher in `framework/src/cost_model/simulation/config/xxx.h`.
3. If you need to add a configuration item, register the new configuration name in the dispatcher function of the corresponding configuration header file.

## F91003 FILE_FORMAT_ERROR

**Error Description**

Failed to parse the JSON file. The content does not conform to JSON syntax specifications.

**Possible Causes**

+ JSON syntax error: missing quotation marks, extra or missing commas, or mismatched brackets.
+ The file encoding is not UTF-8 or contains illegal characters.

   ```json
   # Error example - JSON syntax error (key without quotes, trailing comma)
   {
      key: "value",
      "list": [1, 2,]
   }
   ```

**Solution**

1. Use a JSON validation tool to check the file format:

   ```bash
   # Correct example - validate JSON format
   python3 -m json.tool my_config.json
   ```

2. Ensure that the file encoding is UTF-8.
3. Ensure that all strings in the JSON are enclosed in double quotation marks (`"`) and that there are no trailing commas.

   ```json
   # Correct example - valid JSON
   {
      "key": "value",
      "list": [1, 2]
   }
   ```

## F91004 FILE_CONTENT_ERROR

**Error Description**

The file can be opened and parsed, but its content does not comply with the required field specifications, such as a configuration line missing the `=` delimiter or a numeric field exceeding the valid range.

**Possible Causes**

+ A line in the configuration file does not follow the `key=value` format (missing `=`).

   ```txt
   # Error example - configuration line missing the equal sign
   Pipeline.enable 1
   ```

+ A value in a CSV row exceeds the `uint64_t` range.

   ```txt
   # Error example - value in CSV line exceeds the uint64_t range
   18446744073709551616
   ```

**Solution**

1. Check whether each line in the configuration file follows the `key=value` format, and ensure that the key and value are separated by `=`.

   ```txt
   # Correct example - configuration line in key=value format
   Pipeline.enable=1
   ```

2. Check whether the numeric field is within the valid range (for example, `uint64_t` does not exceed 2^64-1).

## F91005 INVALID_PATH

**Error Description**

The path of the intermediate product files, Python drawing script, or precision simulation target file (`.o`) required for simulation does not exist.

**Possible Causes**

+ The preceding simulation steps (function building, scheduling, etc.) are not completed properly, and intermediate files (`dyn_topo.txt`, `program.json`, `pipe.swim.json`, `swim.json`, `topo.json`, etc.) are not generated.

   ```txt
   # Error example - intermediate file dyn_topo.txt is not generated
   [SIMULATION]: dyn_topo.txt does not exist. topo_txt_path: /path/to/output_xxx/CostModelSimulationOutput/dyn_topo.txt
   ```

+ The path of the Python drawing script (`draw_pipe_swim_lane.py`, `print_swim_lane.py`, `draw_swim_lane.py`) is incorrect.

   ```txt
   # Error example - Python drawing script path does not exist
   [SIMULATION]: draw_pipe_swim_lane.py does not exist. drawScriptPath: /path/to/draw_pipe_swim_lane.py
   ```

+ The precision simulation compilation output (`.o` file) is missing.

   ```txt
   # Error example - compilation output .o file does not exist
   obj file does not exist. objPath: /path/to/output.o
   ```

**Solution**

1. Check the file path reported in the log and verify whether the file exists.

   ```bash
   # Correct example - confirm that the intermediate file has been generated
   ls output_xxx/CostModelSimulationOutput/dyn_topo.txt
   ls output_xxx/CostModelSimulationOutput/program.json
   ```

2. Verify whether all preceding simulation steps have completed successfully.
3. Verify that the Python drawing script is in the expected path. If necessary, copy it from the source directory to the output directory.
4. Verify that the components related to precision simulation have been compiled.

## F91006 FILE_OPEN_FAILED

**Error Description**

Failed to open the specified file, commonly seen with JSON configuration files, calendar files, topology files, etc.

**Possible Causes**

+ The file does not exist or the path is incorrect.

   ```bash
   # Error example - the specified file does not exist
   $ ls non_exist.json
   ls: cannot access 'non_exist.json': No such file or directory
   ```

+ The current user does not have read permission.

   ```bash
   # Error example - the file has no read permission
   $ ls -l my_config.json
   ---------- 1 root root 1024 Jan 1 12:00 my_config.json
   ```

+ The file is corrupted or locked by another process.

**Solution**

1. Verify that the file path is correct.
2. Check whether the current user has read permission on the file:

   ```bash
   # Correct example - confirm the file exists and has read permission
   $ ls -l my_config.json
   -rw-r--r-- 1 user group 1024 Jan 1 12:00 my_config.json
   ```

3. Try to open the file with the corresponding tool to verify its integrity.

## F91007 PYTHON_CMD_ERROR

**Error Description**

During simulation, executing a Python script (such as the swimlane drawing scripts `draw_pipe_swim_lane.py`, `print_swim_lane.py`, and `draw_swim_lane.py`) returns a non-zero exit code.

**Possible Causes**

+ The Python environment is missing dependencies (such as `matplotlib` and `graphviz`).

   ```bash
   # Error example - missing matplotlib dependency
   $ python3 draw_pipe_swim_lane.py input.json
   ModuleNotFoundError: No module named 'matplotlib'
   ```

+ The Python script is incompatible with the current Python version.

   ```bash
   # Error example - Python version incompatibility causes script syntax error
   $ python3 draw_pipe_swim_lane.py input.json
   SyntaxError: invalid syntax
   ```

+ The input file of the script is missing or has an incorrect format.

   ```bash
   # Error example - script input file does not exist
   $ python3 draw_pipe_swim_lane.py missing_input.json
   FileNotFoundError: [Errno 2] No such file or directory: 'missing_input.json'
   ```

**Solution**

1. Check whether the Python environment is available:

   ```bash
   python3 --version
   ```

2. Install the required Python dependencies:

   ```bash
   pip3 install matplotlib graphviz
   ```

   ```bash
   # Correct example - confirm the dependency is installed
   $ python3 -c "import matplotlib; print('OK')"
   OK
   ```

3. Manually execute the Python command that reported the error, view the detailed error output, and fix the issue based on the prompts.

## F92006 INVALID_PIPE_TYPE

**Error Description**

An unrecognized pipeline type is encountered during simulation: the mapping for the corresponding opcode is missing in `SCHED_CORE_PIPE_TYPE`, or `GetAddress()` / `GetSize()` is called on a non-cache type.

**Possible Causes**

+ A new opcode that has not been registered in the `SCHED_CORE_PIPE_TYPE` data structure is used.
+ An address or size query method is called on a non-read-cache/write-cache pipe type.

**Solution**

1. Add the pipe type mapping corresponding to the new opcode in the `SCHED_CORE_PIPE_TYPE` data structure in `framework/src/cost_model/simulation/common/ISA.h`.
2. Ensure that `GetAddress()` / `GetSize()` is called only on cache types.

## F92007 SHAPE_INVALID

**Error Description**

The input tensor shape is empty or the first dimension is empty, so the simulation cannot be executed.

**Possible Causes**

+ The shape of the input tensor is not correctly initialized, or the dynamic shape inference result is empty.

   ```python
   # Error example - shape is an empty list
   x = pypto.tensor([], pypto.DT_FP32)
   ```

**Solution**

1. Check whether the input data shape is valid, and ensure that the shape is not empty and the first dimension is not empty.

   ```python
   # Correct example - shape contains valid dimensions
   x = pypto.tensor([4, 8], pypto.DT_FP32)
   ```

2. For dynamic shape scenarios, verify that the shape inference logic is correct.

## F92010 DEAD_LOCK

**Error Description**

A deadlock is detected during simulation. A machine cannot continue scheduling at a certain cycle.

**Possible Causes**

+ The task dependency graph contains circular dependencies or resource contention.

   ```txt
   # Error example - simulation deadlock log
   [ReportDeadlock] Machine 0 is deadlock at cycle 12345
   Simulation is deadlock at cycle 12345 !!!!!!!!!
   ```

+ The task scheduling logic is defective.

**Solution**

1. Locate the dot file corresponding to the deadlock in the `output/output_xxx/CostModelSimulationOutput/graphs` directory.
2. Use Graphviz to render the dot file to analyze task dependencies:

   ```bash
   dot -Tpng deadlock.dot -o deadlock.png
   ```

3. Check whether the task dependency graph contains circular dependencies or resource contention.
4. If the issue persists, visit the community to submit an [issue](https://gitcode.com/cann/pypto/issues).

## F94001 NO_SO_EXISTS

**Error Description**

The shared library file (such as `libpem_davinci.so`) required for precision simulation failed to load. The file does not exist or the path is incorrect.

**Possible Causes**

+ The precision simulation .so file is not compiled or not installed to the expected path.
+ The compilation output path is incorrectly configured.

   ```txt
   # Error example - Failed to load the shared library file
   can not load library: /path/to/libpem_davinci.so
   ```

**Solution**

1. Enable simulation logs to locate the missing .so path:

   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

2. Search the logs for `can not load library:` to find the missing .so file path.
3. Verify that the precision simulation-related components have been compiled and installed.

   ```bash
   # Correct example - Confirm that the .so file exists
   ls -l /path/to/libpem_davinci.so
   ```

## F94002 CANN_LOAD_FAILED

**Error Description**

The CANN environment is not correctly loaded, the `ASCEND_HOME_PATH` environment variable is not set, and precision simulation is unavailable.

**Possible Causes**

+ The CANN environment initialization script has not been executed.

   ```bash
   # Error example - ASCEND_HOME_PATH is not set
   $ echo $ASCEND_HOME_PATH
   (empty)
   ```

**Solution**

1. Execute the CANN environment initialization script:

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. Verify that the environment variable has been set:

   ```bash
   # Correct example - environment variable is set
   $ source /usr/local/Ascend/ascend-toolkit/set_env.sh
   $ echo $ASCEND_HOME_PATH
   /usr/local/Ascend/ascend-toolkit/latest
   ```

## F94003 CMD_ERROR

**Error Description**

An external command failed during precision simulation, for example, command string formatting truncation caused `snprintf_s` to fail, or `llvm-objcopy` returned a non-zero exit code.

**Possible Causes**

+ The command string is too long, causing formatting truncation.
+ The `llvm-objcopy` tool is not installed or its version is incompatible.

   ```bash
   # Error example - llvm-objcopy not installed
   $ llvm-objcopy --version
   bash: llvm-objcopy: command not found
   ```

+ The target file path is abnormal, causing command execution failure.

   ```txt
   # Error example - target file path is abnormal
   cmd error: llvm-objcopy --only-section=.text /path/to/source.o /path/to/nonexistent/target.o
   ```

**Solution**

1. Check whether the complete command string printed in the log is correct.
2. Verify that the `llvm-objcopy` tool is installed and its version is compatible:

   ```bash
   llvm-objcopy --version
   ```

3. Manually run the command from the log in the terminal to view the specific error output.

## F94005 CANNSIM_FAILED

**Error Description**

The precision simulation was not started through `cannsim`, making the precision simulation unavailable.

**Possible Causes**

+ Precision simulation was not started using the `cannsim record` command.

   ```bash
   # Incorrect example: started directly with python3 instead of cannsim
   python3 my_script.py --run_mode sim
   ```

**Solution**

1. Start precision simulation using the `cannsim record` method:

   ```bash
   # Correct example: start the precision simulation through cannsim
   cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950
   ```

## F9FFFF SIM_INNER_ERROR

**Error Description**

NA

**Possible Causes**

NA

**Solution**

1. Visit the community and submit an [issue](https://gitcode.com/cann/pypto/issues).
