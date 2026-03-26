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

from unittest_utils import rtol_atol, DeviceGuard


paddle.set_device("musa")

python_api = paddle.eye


class TestEyeOp(unittest.TestCase):
    def test_eye_op_forward_all_dtypes(self):
        for dtype in ("float32", "int", "int64", "float64"):
            with self.subTest(dtype=dtype):
                out = python_api(5).astype(dtype)

                with DeviceGuard():
                    golden = python_api(5).astype(dtype)

                rtol, atol = rtol_atol(dtype, is_grad=False)
                self.assertTrue(
                    paddle.allclose(out, golden, rtol=rtol, atol=atol),
                    msg=f"forward mismatch for {dtype}",
                )
