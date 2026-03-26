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


class TestAdamaxOp(unittest.TestCase):
    def test_adamax_op_forward_all_dtypes(self):
        shape = (3, 4, 5)
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                inputs = prepare_inputs(shape, dtype)
                x = np_to_tensor(inputs, dtype, stop_gradient=False)

                optimizer = paddle.optimizer.Adamax(learning_rate=0.1, parameters=[x])
                y = x**2
                y.backward()

                optimizer.step()
                out = x

                with DeviceGuard():
                    x_cpu = np_to_tensor(inputs, dtype, stop_gradient=False)
                    optimizer = paddle.optimizer.Adamax(
                        learning_rate=0.1, parameters=[x_cpu]
                    )
                    y_cpu = x_cpu**2
                    y_cpu.backward()

                    optimizer.step()
                    golden = x_cpu

                rtol, atol = rtol_atol(dtype, is_grad=True)
                self.assertTrue(
                    np.allclose(out, golden, rtol=rtol, atol=atol),
                    msg=f"backward mismatch for {dtype}",
                )
