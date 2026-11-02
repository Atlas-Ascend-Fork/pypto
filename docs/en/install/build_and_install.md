# PyPTO Installation

> **Note**
>
> 1. If you are using a version **earlier than CANN 9.1.0**, download the PyPTO source code that matches the CANN version, and refer to `build_and_install.md` on the corresponding branch to compile and install PyPTO. For the mapping between CANN versions and PyPTO versions, see the following table:
>
>    | CANN Version | PyPTO Version |
>    | :--- | :--- |
>    | 8.5.0 | 0.1.2 |
>    | 9.0.0 | 0.2.0 |
>
> 2. If you are using **CANN 9.1.0 or later**, PyPTO is integrated into the CANN package. You can use PyPTO after the CANN installation is complete and can skip this section.
> 3. If you want to experience the capabilities of the latest PyPTO master version or develop the PyPTO framework based on the source code, refer to this document to compile and install PyPTO from source.

## Downloading the Source Code

Download the source code of the corresponding branch based on the CANN software version. \$\{tag\_version\} indicates the branch tag name.

```bash
# Download the source code of the corresponding project branch
git clone -b ${tag_version} https://gitcode.com/cann/pypto.git
```

In the WebIDE environment, **the project source code of the latest commercial release is provided by default**. To obtain the source code of another version, download it by running the preceding command.

> [!NOTE] Note
>
> - When the GitCode platform uses HTTPS, you need to configure and use a personal access token instead of the login password for operations such as cloning and pushing.
> - If the compilation environment cannot access the network and the code cannot be downloaded by running the git command, download the source code in a network-connected environment and then manually upload it.

## Building and Installing from Source

### Building the run package

Go to the root directory of the PyPTO source code and run the following command to build the run package:

```bash
python3 build_ci.py --clean --py_abi=37 --plat_name=manylinux2014 --no_isolation --whl_into_run
```

**Parameter description**:

| Parameter | Description |
| :--- | :--- |
| `--clean` | Cleans the build directory and installation output directory before compilation. |
| `--py_abi` | Specifies the numeric part of the Python ABI tag of the whl package. For example, `37` corresponds to `cp37`. |
| `--plat_name` | Specifies the platform tag of the whl package, for example, `manylinux2014`. The build script generates the complete platform information based on the current system architecture. |
| `--no_isolation` | Disables the whl isolated build mode. Build dependencies must be installed in the current environment in advance. |
| `--whl_into_run` | Packs the built whl package into the run installation package. |

### Installation

After you run the preceding compilation command, a run package is generated in the `build_out` directory of the PyPTO source code root directory. The file name is similar to `cann-pypto_9.1.0_linux-aarch64.run`. Go to the `build_out` directory and run the following command to complete the installation:

```bash
cd build_out
bash ./cann-pypto_${pypto_version}_${os_arch}.run --full -q --pylocal
```

Variable description:

- \$\{pypto_version}: indicates the version number of the PyPTO package, for example, `9.1.0`.
- \$\{os_arch}: indicates the operating system and CPU architecture, for example, `linux-aarch64`.
- `--full`: indicates a full installation.
- `-q`: indicates a silent installation.
- `--pylocal`: specifies whether to install Python-related information to the installation path of the CANN software package during package installation.

## Installation Verification

After the preceding steps are complete, refer to [Running Samples](../invocation/examples_invocation.md) to run related test cases and check whether PyPTO has been installed successfully.
