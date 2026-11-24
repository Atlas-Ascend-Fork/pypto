# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(pypto_extract_aicpu_tar base_dir whl_glob)
    set(_pypto_aicpu_tar_dir "${base_dir}/opp/built-in/op_impl/aicpu/kernel")
    file(GLOB _pypto_whl_list "${whl_glob}")
    file(REMOVE "${_pypto_aicpu_tar_dir}/pypto_aicpu.tar.gz")
    list(LENGTH _pypto_whl_list _pypto_whl_count)
    if(_pypto_whl_count GREATER 0)
        list(GET _pypto_whl_list 0 _pypto_whl_file)
        execute_process(
            COMMAND python3 -c "import os,sys,zipfile\nwhl,out=sys.argv[1],sys.argv[2]\nwith zipfile.ZipFile(whl) as z:\n    names=[n for n in z.namelist() if n.endswith('pypto_aicpu.tar.gz')]\n    if names:\n        os.makedirs(out,exist_ok=True)\n        open(os.path.join(out,'pypto_aicpu.tar.gz'),'wb').write(z.read(names[0]))" "${_pypto_whl_file}" "${_pypto_aicpu_tar_dir}"
            RESULT_VARIABLE _pypto_aicpu_ret
            ERROR_VARIABLE _pypto_aicpu_err
        )
        if(NOT _pypto_aicpu_ret EQUAL 0)
            message(FATAL_ERROR "extract pypto_aicpu.tar.gz from ${_pypto_whl_file} failed: ${_pypto_aicpu_err}")
        endif()
    endif()
endfunction()
