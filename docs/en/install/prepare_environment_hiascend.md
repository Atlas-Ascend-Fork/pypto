# Environment Setup<a name="ZH-CN_TOPIC_0000002532541127"></a>

<!-- md-trans-meta sourceCommit=a785a3f1b1151fe8cda50954a981f63ea65b251a translatedAt=2026-09-10T11:25:38.321Z pushedAt=2026-09-11T02:52:31.733Z -->

- Before developing or running PyPTO, install the **driver firmware** and **CANN software package**. For details, see [CANN Quick Installation](https://www.hiascend.com/cann/download).

    > [!NOTE] Note
    >  To use the CANN running user for compilation and running after CANN software package installation, log in to the environment as the CANN running user and run the `source ${INSTALL_DIR}/set_env.sh` command to set the environment variables. Replace `${INSTALL_DIR}` with the storage path of the CANN software after installation. For example, if the software is installed by the **root** user, the default storage path after installation is `/usr/local/Ascend/cann`.

- Install the Python dependencies of PyPTO.

    - Python: version >= 3.9.
        - Install the Python development component (commonly known as `python3-dev`).

    - Install the Python dependency packages.

        The required pip packages and their versions are described in `python/requirements.txt`. You can install them by running the following commands:

        ```bash
        # Enter the root directory of the PyPTO project source code.
        cd pypto

        # Install the related pip package dependencies.
        python3 -m pip install -r python/requirements.txt
        ```

    - Install PyTorch and TorchNPU.

        Ensure that the CANN toolkit package is installed before installing `TorchNPU`. Install it separately based on the Python version of the actual environment. For detailed instructions, see the *software installation* manual in the [TorchNPU documentation center](https://hiascend.com/document/redirect/pytorchuserguide). Ensure that `PyTorch`, `TorchNPU`, and `PyPTO` use the same Python version.

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
    - pybind11 >= 2.13.6 (pip package, installable by running `python3 -m pip install pybind11`)

- Install PyPTO Toolkit.

For usage details, see the [PyPTO Toolkit documentation](https://pypto-tools.gitcode.com/index.html)

> [!NOTE] Note
> For PyPTO development, installing the driver firmware is not mandatory. On non-Ascend devices, you can use the CPU simulation environment to develop and test PyPTO first, and then use Ascend devices for verification and accelerated computation once everything is ready. The simulation feature is currently an experimental feature and may change in later versions. It is not supported in production environments. For installation on non-Ascend devices, see "Appendix B: Common Operations > Installing Software Packages on a Non-Ascend Device" in the [*CANN installation guide*](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware).
