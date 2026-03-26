# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import paddle


def prepare_inputs(shape, dtype):
    if dtype == "int":
        arr = np.random.randint(-3, 4, size=shape, dtype=np.int32)
        return arr
    elif dtype == "int64":
        arr = np.random.randint(-3, 4, size=shape, dtype=np.int64)
        return arr
    elif dtype in ("float16", "float32", "bfloat16", "float64"):
        arr = (np.random.random(shape).astype(np.float32) - 0.5) * 2
        # randomly set 0 to test grad at 0
        mask = np.random.random(shape) < 0.2
        arr[mask] = 0.0
        if dtype == "float16":
            return arr.astype(np.float16)
        elif dtype == "float32":
            return arr.astype(np.float32)
        elif dtype == "float64":
            return arr.astype(np.float64)
        else:
            return arr.astype(np.float32)
    elif dtype == "complex":
        real = (np.random.random(shape).astype(np.float32) - 0.5) * 2
        imag = (np.random.random(shape).astype(np.float32) - 0.5) * 2
        # randomly set 0 to test grad at 0
        mask = np.random.random(shape) < 0.2
        real[mask] = 0.0
        imag[mask] = 0.0
        return (real + 1j * imag).astype(np.complex64)
    elif dtype == "bool":
        arr = np.ones(shape)
        mask = np.random.random(shape) < 0.5
        arr[mask] = 0
        return arr.astype(bool)
    else:
        raise ValueError(dtype)


#TODO(yuanyifu): add fp8
def np_to_tensor(arr, dtype, stop_gradient=True):
    if dtype == "bfloat16":
        return paddle.to_tensor(arr, dtype=paddle.bfloat16, stop_gradient=stop_gradient)
    dtype_map = {
        "int": paddle.int32,
        "int64": paddle.int64,
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "complex": paddle.complex64,
        "bool": paddle.bool,
    }
    return paddle.to_tensor(arr, dtype=dtype_map[dtype], stop_gradient=stop_gradient)


def rtol_atol(dtype, is_grad=False):
    if dtype == "float16" or dtype == "bfloat16":
        return (1e-2, 1e-2) if not is_grad else (2e-2, 2e-2)
    if dtype == "complex":
        return (1e-6, 1e-6) if not is_grad else (1e-5, 1e-5)
    if dtype == "int" or dtype == "int64":
        return (0.0, 0.0)
    return (1e-6, 1e-6) if not is_grad else (1e-5, 1e-5)


class DeviceGuard:
    def __init__(self, enter_device="cpu", exit_device="musa"):
        self.enter_device = enter_device
        self.exit_device = exit_device

    def __enter__(self):
        paddle.set_device(self.enter_device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.set_device(self.exit_device)
