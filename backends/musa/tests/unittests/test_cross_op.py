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

python_api = paddle.cross


class TestCrossOp(unittest.TestCase):
    def test_cross_op_forward_all_dtypes(self):
        shape = (3, 4, 5)
        for dtype in ("float32", "float64", "int", "int64"):
            with self.subTest(dtype=dtype):
                inputs1 = prepare_inputs(shape, dtype)
                inputs2 = prepare_inputs(shape, dtype)

                x1 = np_to_tensor(inputs1, dtype, stop_gradient=True)
                x2 = np_to_tensor(inputs2, dtype, stop_gradient=True)
                out = python_api(x1, x2)

                with DeviceGuard():
                    x1_cpu = np_to_tensor(inputs1, dtype, stop_gradient=True)
                    x2_cpu = np_to_tensor(inputs2, dtype, stop_gradient=True)
                    golden = python_api(x1_cpu, x2_cpu)

                rtol, atol = rtol_atol(dtype, is_grad=False)
                self.assertTrue(
                    paddle.allclose(out, golden, rtol=rtol, atol=atol),
                    msg=f"forward mismatch for {dtype}",
                )

    def test_cross_op_backward_all_dtypes(self):
        shape = (3, 4, 5)
        # TODO: gaussian kernel implementation for int cross op
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                inputs1 = prepare_inputs(shape, dtype)
                inputs2 = prepare_inputs(shape, dtype)

                x1 = np_to_tensor(inputs1, dtype, stop_gradient=False)
                x2 = np_to_tensor(inputs2, dtype, stop_gradient=False)
                n = paddle.randn(shape, dtype=dtype)
                y = (python_api(x1, x2) * n).sum()
                y.backward()

                grad = x1.grad

                with DeviceGuard():
                    x1_cpu = np_to_tensor(inputs1, dtype, stop_gradient=False)
                    x2_cpu = np_to_tensor(inputs2, dtype, stop_gradient=False)
                    n_cpu = n.to("cpu")
                    y_cpu = (python_api(x1_cpu, x2_cpu) * n_cpu).sum()
                    y_cpu.backward()
                    golden = x1_cpu.grad

                rtol, atol = rtol_atol(dtype, is_grad=True)
                self.assertTrue(
                    np.allclose(grad, golden, rtol=rtol, atol=atol),
                    msg=f"backward mismatch for {dtype}",
                )
