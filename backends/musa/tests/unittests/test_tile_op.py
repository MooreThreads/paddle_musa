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

from __future__ import print_function

import unittest
import paddle

import numpy as np

from unittest_utils import prepare_inputs, np_to_tensor, rtol_atol, DeviceGuard


paddle.set_device("musa")

python_api = paddle.tile


class TestTlieOp(unittest.TestCase):
    def test_tile_op_forward_all_dtypes(self):
        shape = (3, 4, 5)
        for dtype in ("float32", "int", "int64", "float64", "bool"):
            with self.subTest(dtype=dtype):
                inputs = prepare_inputs(shape, dtype)

                x = np_to_tensor(inputs, dtype, stop_gradient=True)
                out = python_api(x, 3)

                with DeviceGuard():
                    x_cpu = np_to_tensor(inputs, dtype, stop_gradient=True)
                    golden = python_api(x_cpu, 3)

                rtol, atol = rtol_atol(dtype, is_grad=False)
                self.assertTrue(
                    paddle.allclose(out, golden, rtol=rtol, atol=atol),
                    msg=f"forward mismatch for {dtype}",
                )
