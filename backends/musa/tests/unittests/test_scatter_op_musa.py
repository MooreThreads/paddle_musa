#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    get_places,
    is_custom_device,
)
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((10, 50)).astype(target_dtype)
        updates_np = np.random.random((10, 50)).astype(target_dtype)

        index_np = np.random.choice(
            np.arange(ref_np.shape[0]),
            size=(updates_np.shape[0],),
            replace=False,
        ).astype("int32")

        # randomly mapping index into equivalent negative index(mod ref_np.shape[0])
        # to test for negative index
        random_negative_mask = (np.random.rand(index_np.shape[0]) > 0.5).astype(
            "bool"
        )
        index_np[random_negative_mask] -= ref_np.shape[0]

        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ["X", "Updates"],
            "Out",
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            max_relative_error=0.008,
        )


class TestScatterFP16Op(TestScatterOp):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestScatterBF16Op(TestScatterOp):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )


class TestScatterOp0(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self._set_dtype()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ["X", "Updates"],
            "Out",
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestScatterFP16Op0(TestScatterOp0):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestScatterBF16Op0(TestScatterOp0):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )


class TestScatterOp1(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        zeros_np = np.zeros([2, 3]).astype(target_dtype)
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ["X", "Updates"],
            "Out",
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestScatterNegativeAxis(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.dtype = np.float32
        target_dtype = "float16" if self.dtype == np.float16 else "float32"

        ref_np = np.ones((3, 3)).astype(target_dtype)
        zeros_np = np.zeros([2, 3]).astype(target_dtype)
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)

        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]

        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)

        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda() or is_custom_device():
            places.append(get_device_place())
        for place in places:
            self.check_output_with_place(place)

    def test_check_grad(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda() or is_custom_device():
            places.append(get_device_place())
        for place in places:
            self.check_grad_with_place(
                place,
                ["X", "Updates"],
                "Out",
            )


class TestOutOfRangeError(unittest.TestCase):
    def test_dygraph_forward(self):
        with dygraph_guard():
            _ = paddle.scatter(
                x=paddle.randn([100, 3]).cpu(),
                index=paddle.to_tensor([0, 99, -100]).cpu(),
                updates=paddle.randn([3, 3]).cpu(),
                overwrite=False,
            )

    def test_dygraph_error(self):
        with dygraph_guard():
            # out of lower bound
            with self.assertRaises(IndexError):
                _ = paddle.scatter(
                    x=paddle.randn([100, 3]).cpu(),
                    index=paddle.to_tensor([0, 99, 100]).cpu(),
                    updates=paddle.randn([3, 3]).cpu(),
                    overwrite=False,
                )
            # out of upper bound
            with self.assertRaises(IndexError):
                _ = paddle.scatter(
                    x=paddle.randn([100, 3]).cpu(),
                    index=paddle.to_tensor([0, 99, -101]).cpu(),
                    updates=paddle.randn([3, 3]).cpu(),
                    overwrite=False,
                )


class TestScatterFP16Op1(TestScatterOp1):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestScatterOp5(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_output_with_place(
                place, atol=1e-3, check_pir=True, check_symbol_infer=False
            )

    def test_check_grad(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )


class TestScatterOp_ZeroSize(OpTest):
    def setUp(self):
        paddle.disable_static()
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self._set_dtype()
        ref_np = np.ones((100, 1)).astype(self.dtype)
        updates_np = np.random.random((4, 1)).astype(self.dtype)
        index_np = np.random.random([0]).astype("int32")

        output_np = np.copy(ref_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_pir=True,
            max_relative_error=0.008,
        )


class TestScatterOp_ZeroSize2(TestScatterOp_ZeroSize):
    def setUp(self):
        paddle.disable_static()
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self._set_dtype()
        ref_np = np.ones((0, 1)).astype(self.dtype)
        updates_np = np.random.random((4, 1)).astype(self.dtype)
        index_np = np.random.random([4]).astype("int32")

        output_np = np.copy(ref_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_grad(self):
        self.check_grad(
            ["X", "Updates"],
            "Out",
            check_pir=True,
            max_relative_error=0.008,
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
