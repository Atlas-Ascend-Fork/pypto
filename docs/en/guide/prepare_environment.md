# Preparing the Environment

<!-- md-trans-meta sourceCommit=0950534231f8a8e7080cc299d29bbb6f7e5591c5 translatedAt=2026-11-15T01:33:09.250Z pushedAt=2026-11-18T06:03:01.294Z nl=2 -->

- Before developing or running PyPTO, install the **driver firmware** and **CANN software package**. For details, see [*CANN Quick Installation*](https://www.hiascend.com/cann/download) to complete environment preparation.

    > [!NOTE] Note
    > After installing the CANN software package, when you compile and run as the CANN running user, log in to the environment as the CANN running user and run the `source ${INSTALL_DIR}/set_env.sh` command to set environment variables. Replace `${INSTALL_DIR}` with the storage path of the installed CANN software. For example, if you install the software as the root user, the default storage path is `/usr/local/Ascend/cann`.

- Install the Python dependencies of PyPTO.

    - Python: version >= 3.9.
        - Install the Python development component (commonly known as `python3-dev`).

    - Install the Python dependency packages.

        The required pip packages and their versions are described in `python/requirements.txt`. You can run the following command to install them:

        ```bash
        # Enter the root directory of the PyPTO project source code.
        cd pypto

        # Install the required pip package dependencies.
        python3 -m pip install -r python/requirements.txt
        ```

    - Install PyTorch and TorchNPU.

        Make sure to install the CANN toolkit package before installing `TorchNPU`. Install it separately based on the Python version of your actual environment. For detailed instructions, see [TorchNPU Software Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download). Ensure that `PyTorch`, `TorchNPU`, and `PyPTO` use the same Python version.

- Install CMake. PyPTO requires CMake 3.16.3 or later. If the version does not meet the requirement, install a compliant version by referring to the following example.

    Example: Install CMake 3.16.3 (x86_64 architecture).

    ```bash
    mkdir -p cmake-3.16 && wget -qO- "https://cmake.org/files/v3.16/cmake-3.16.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C cmake-3.16
    export PATH=`pwd`/cmake-3.16/bin:$PATH
    ```

- Install other dependencies.

    - make
    - g++ >= 7.3.1
    - gcc >= 7.3.1
    - pybind11 >= 2.13.6 (pip package, which can be installed by running `python3 -m pip install pybind11`)

- Install PyPTO Toolkit.

For usage documentation, see [PyPTO Toolkit Documentation](https://pypto-tools.gitcode.com/index.html).

> [!NOTE] Note
> For PyPTO development, installing the driver firmware is not mandatory. On non-Ascend devices, you can use the CPU emulation environment to develop and test PyPTO first, and then use Ascend devices for verification and accelerated computation once everything is ready. The emulation feature is currently an experimental feature and may change in later versions. It is not supported for use in production environments. For installation on non-Ascend devices, see the "Appendix B: Common Operations > Installing CANN on Non-Ascend Devices" section in [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum).
