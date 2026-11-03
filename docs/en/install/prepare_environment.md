# Environment Deployment

Before developing or running PyPTO, complete the basic environment setup, compilation, and installation by following the steps below. Ensure that the NPU driver and firmware and the CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) are installed.

## Environment Installation

This project provides multiple methods for setting up an Ascend environment. Select one as required.

> **Note**: The compilation state and running state mentioned in this document are defined as follows. Select one based on your actual situation.
>
> - Compilation state: For scenarios where PyPTO is only compiled but not run, you need to install only the CANN toolkit package and the compilation dependencies of PyPTO.
> - Running state: For scenarios where PyPTO is run (compile-and-run or run only), you need to install the driver and firmware, the CANN toolkit package, and the CANN ops package.

|  Installation Method  |  Description  |  Scenario  |
| ----- | ------ | ------ |
|  CANNLab  | A one-stop development platform that provides an online Ascend environment which can run directly, without manual installation.<br>It currently provides single-node compute capability and **installs the latest commercial CANN package by default**. | Suitable for developers without Ascend devices. |
|  Docker  | The CANN image has CANN and the dependencies required for running PyPTO pre-integrated and is ready to use out of the box.<br>The environment installs the latest commercial CANN package by default. When downloading the source code, ensure that it matches the software. | Suitable for developers who have Ascend devices and **need to quickly set up an environment**. |
|  Manual installation  | Manually install the CANN packages and basic dependencies of PyPTO. This method provides high flexibility. | Suitable for developers who have Ascend devices and **want to experience the capabilities of the latest PyPTO master branch or released versions, or develop the PyPTO framework based on the source code**. |

### Method 1: CANNLab

For developers without Ascend devices, you can directly use the CANNLab cloud development environment, that is, the "**one-stop development platform**". This platform provides an online Ascend environment that can run directly. The required driver and firmware, software packages, and dependencies have been installed in the environment, so no manual installation is required.

> **Note**: The environment installs the latest commercial CANN package by default. When downloading the source code, ensure that it matches the software. For more information about the development platform, refer to the [CANNLab guide](https://gitcode.com/org/cann/cannlab/docs?code=quick-start-guide).

### Method 2: Docker Deployment

For developers with Ascend devices, if you want to quickly set up an Ascend environment, use a Docker image deployment.

> **Note**:
>
> - The image file is large, so the download takes some time. Wait patiently. For details about the options of the docker command, run `docker --help`.
> - The environment installs the latest commercial CANN package by default. When downloading the source code, ensure that it matches the software.

1. **Install the driver and firmware (dependency in the running state)**

    The driver and firmware are dependencies in the running state and do not need to be installed if you only compile PyPTO. Run `npu-smi info` to check whether NPU information is displayed. If not, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the driver and firmware.

2. **Download the image**

    - Step 1: Log in to the host machine as the root user. Ensure that Docker Engine (version 1.11.2 or later) is installed on the host machine. Run `docker --version` to check the Docker version. If Docker is not installed, refer to the [Docker official installation guide](https://docs.docker.com/engine/install/).
    - Step 2: Pull an image that has the CANN software package and the dependencies required for running PyPTO pre-integrated from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884).

        The following is an example. Replace the CANN version, chip series, operating system, and Python version information as required.

        ```bash
        # Use cann:9.1.0-beta.1 as an example
        docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel
        ```

3. **Run Docker**

    After the image is pulled, start the container with specific parameters so that the Ascend devices on the host machine can be accessed in the container.

    ```bash
    docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel bash
    ```

    | Parameter | Description | Precautions |
    | :--- | :--- | :--- |
    | `--name cann_container` | Specifies a name for the container for easy management. | Customizable. |
    | `--device /dev/davinci0` | Core: maps the NPU device cards on the host machine to the container. Multiple NPU device cards can be mapped. | Must be adjusted based on the actual situation: `davinci0` corresponds to NPU card 0 in the system. Run the `npu-smi info` command on the host machine first, and change the number based on the device numbers (such as `NPU 0` and `NPU 1`) in the command output.|
    | `--device /dev/davinci_manager` | Maps the NPU device management interface. | - |
    | `--device /dev/devmm_svm` | Maps the device memory management interface. | - |
    | `--device /dev/hisi_hdc` | Maps the communication interface between the host and the device. | - |
    | `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts the tools and libraries related to the device container management interface (DCMI). | - |
    | `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mounts the `npu-smi` tool. | Allows this command to be run directly in the container to query the NPU status and performance information.|
    | `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: maps the NPU driver library on the host machine to the container. | - |
    | `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mounts the driver version information file. | - |
    | `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mounts the CANN software installation information file. | - |
    | `-it` | A combination of `-i` (interactive) and `-t` (allocating a pseudo terminal). | - |
    | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel` | Specifies the Docker image to run. | Ensure that this image name and tag are exactly the same as those of the image you pulled by running `docker pull`. |
    | `bash` | Command executed immediately after the container starts. | - |

4. **Install PyPTO dependencies**

    After entering the container, refer to [Manual Installation - PyPTO Dependencies](#pypto-dependencies) to install the Python dependencies and other compilation dependencies.

### Method 3: Manual Installation

For developers with Ascend devices, if you want to manually set up an Ascend environment, follow the steps below.

#### Installing the Software

- **Scenario 1: Experience the capabilities of the master version or perform development based on the master version**

    1. **Install the driver and firmware (dependency in the running state)**

        The driver and firmware are dependencies in the running state and do not need to be installed if you only compile PyPTO. Run `npu-smi info` to check whether NPU information is displayed. If not, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the driver and firmware.

        > **Important**:
        >
        > - Supported versions: Ascend HDK 25.5.1 or later.
        > - HDK environments earlier than the supported versions are not within the scope of PyPTO verification and support. Running an abnormal PyPTO program may cause an abnormal NPU status, which further affects subsequent task execution and causes problems such as AIC timeouts. In severe cases, the device or host machine must be restarted for recovery.

    2. **Install the CANN packages**

        Click [Download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the version with the latest timestamp, and download the corresponding packages based on the product model and environment architecture. The installation commands are as follows. For more instructions, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download).

        - Install the CANN toolkit package.

            ```bash
            bash ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - Install the CANN ops package (dependency in the running state)

            The ops package is a dependency in the running state and does not need to be installed if you only compile PyPTO.

            ```bash
            bash ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        Variable description:

        - \$\{cann\_version\}: indicates the version number of the CANN package.
        - \$\{arch\}: indicates the CPU architecture, which can be queried by running `uname -m`, for example, aarch64 or x86_64.
        - \$\{soc\_name\}: indicates the NPU model name.
        - \$\{install\_path\}: indicates the specified installation path. The ops package must be installed in the same path as the toolkit package. For the root user, the default installation directory is `/usr/local/Ascend`.

        <!-- npu="x90,9030,9050" id2 -->
        Kirin processor products: only the CANN toolkit package is required, and the CANN ops package does not need to be installed.
        <!-- end id2 -->

- **Scenario 2: Experience the capabilities of a released PyPTO version or perform development based on a released version**

    Visit the [CANN official website download center](https://www.hiascend.com/cann/download), select the released CANN version that matches the PyPTO version, download the corresponding packages based on the product model and environment architecture, and run the commands provided on the web page to complete the installation.

#### PyPTO Dependencies

1. **Install Python dependencies**

    - Python: version 3.9 or later
        - **Important**: The Development component of Python (usually called `python3-dev`) must be installed.

    - Install the Python dependency packages:

        The dependent pip packages and their versions are described in `python/requirements.txt`. You can run the following command to complete the installation:

        ```bash
        # Go to the root directory of the PyPTO project source code
        cd pypto

        # Install the related pip package dependencies
        python3 -m pip install -r python/requirements.txt
        ```

    - PyTorch and TorchNPU:
        - **Order**: Install `TorchNPU` only after the toolkit package described in the preceding "Installing the CANN Packages" section is installed.
        - Install them separately based on the Python version of the actual environment. For detailed instructions, refer to [TorchNPU Software Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download).
        - **Important**: Ensure that `PyTorch`, `TorchNPU`, and `PyPTO` use the same Python version.
        <!-- npu="x90,9030,9050" id1 -->
        Kirin processor products: only PyTorch is required, and TorchNPU does not need to be installed.
        <!-- end id1 -->

2. **Install other dependencies**

    - cmake >= 3.16.3
    - make
    - g++ >= 7.3.1
    - gcc >= 7.3.1
    - pybind11 >= 2.13.6 (a pip package that can be installed by running `python3 -m pip install pybind11`)
    - patch

## Environment Verification

After the CANN packages are installed, verify that the environment and driver work properly.

- **Check NPU devices**

    ```bash
    # Run npu-smi. If the device information is properly displayed, the driver works properly
    npu-smi info
    ```

- **Check the CANN version**

    ```bash
    # View the version information of the CANN toolkit and ops packages (installed in the default path). In the CANNLab scenario, replace /usr/local with /home/developer
    cat /usr/local/Ascend/cann/${arch}-linux/ascend*install.info
    ```

    In the command, \${arch} indicates the current architecture, which can be queried by running `uname -m`, for example, aarch64 or x86_64.

After the environment is ready, refer to the [PyPTO Installation](./build_and_install.md) document to install PyPTO.

## Optional Installation

### PyPTO Toolkit Plugin

To use the compute graph and swimlane diagram viewing capabilities, install the PyPTO Toolkit plugin:
For detailed usage documentation, refer to the [PyPTO Toolkit documentation](https://pypto-tools.gitcode.com/index.html)

### MPI Dependencies

The distributed samples of PyPTO depend on MPI. Version 3.2.1 or later is recommended.

**Method 1: Install by using a system package manager**

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y mpich

# CentOS/RHEL
yum install -y mpich
```

**Method 2: Build and install from the source code**

```bash
# Use version 3.2.1 as an example
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

After the installation is complete, set the environment variables:

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

## Configuring Environment Variables

Select a proper command as required to make the CANN environment variables take effect. The preceding environment variable configuration takes effect only in the current window. You can add the commands to an environment variable configuration file (such as the `.bashrc` file) as required.

```bash
# Installed in the default path as the root user (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh

# Installed in a specified path
source ${install_path}/cann/set_env.sh
```
