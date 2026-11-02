# Overview

<!-- md-trans-meta sourceCommit=c757335981cdfedb9fcaea5bed6c066a8dc94f1f translatedAt=2026-10-27T01:31:37.187Z pushedAt=2026-10-27T03:56:34.144Z nl=2 -->

This document describes the error codes of the PyPTO framework and provides troubleshooting guidance.

- F0XXXX indicates an issue with external usage. Directly refer to the error message printed on the screen. If the error message is unclear, visit the community and submit an [Issue](https://gitcode.com/cann/pypto/issues).
- Non-F0XXXX indicates an internal framework issue. Refer to the troubleshooting guidance of the corresponding component based on the error code range. If no matching error code is found or the issue remains unresolved, visit the community and submit an [Issue](https://gitcode.com/cann/pypto/issues).
- Before troubleshooting, configure the following common log environment variables to quickly obtain key logs. For details, see [*Environment Variable Reference*](https://gitcode.com/cann/docs/blob/master/docs/en/env-vars/README.md).

    ```bash
    # Example
    export ASCEND_GLOBAL_LOG_LEVEL=0
    export ASCEND_HOST_LOG_FILE_NUM=1000
    export ASCEND_WORK_PATH=./wk
    ```
