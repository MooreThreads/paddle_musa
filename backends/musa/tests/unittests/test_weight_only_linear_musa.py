# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import math
import os
import re
import struct
import unittest

import numpy as np

import paddle
import paddle.nn.quant as Q
from paddle import base
from paddle.base import core
from paddle.framework import set_default_dtype
from paddle.pir_utils import IrGuard

np.random.seed(123)
paddle.seed(123)


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)



class WeightOnlyLinearTestCase(unittest.TestCase):
    def config(self):
        self.dtype = 'float16'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.batch = 1
        self.token = 32
        self.in_features = 64
        self.out_features = 256
        self.weight_dtype = "int4"
        self.static = False
        self.group_size = -1

    def weightQuantizeCPUGPUConsistenceCheck(self, weight_float):
        for arch in [70, 75, 80, 86]:
            weight_gpu, weight_scale_gpu = Q.weight_quantize(
                (
                    weight_float.to("musa:0")
                    if self.weight_dtype == "int4"
                    else self.weight.to("musa:0")
                ),
                algo=(
                    "weight_only_int8"
                    if self.weight_dtype == "int4"
                    else "weight_only_int4"
                ),
                arch=arch,
                group_size=self.group_size,
            )
            weight_cpu, weight_scale_cpu = Q.weight_quantize(
                weight_float.to("musa:0"),
                algo=(
                    "weight_only_int8"
                    if self.weight_dtype == "int4"
                    else "weight_only_int4"
                ),
                arch=arch,
                group_size=self.group_size,
            )
            np.testing.assert_allclose(
                weight_gpu.numpy(),
                weight_cpu.numpy(),
                atol=1.5,
                rtol=2,
            )
            np.testing.assert_allclose(
                weight_scale_gpu.numpy(),
                weight_scale_cpu.numpy(),
                atol=1e-5,
                rtol=1e-3,
            )
            pass
        pass

    def setUp(self):
        self.config()
        if self.dtype == "bfloat16" or self.weight_dtype == "int4":
            self.atol = 1.4e-1
        x = np.random.random((self.batch, self.token, self.in_features))
        self.x = paddle.to_tensor(x, dtype=self.dtype)
        if self.bias:
            bias_attr = base.ParamAttr(
                trainable=False,
                regularizer=None,
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
        else:
            bias_attr = None
        set_default_dtype(self.dtype)
        self.linear = paddle.nn.Linear(
            self.in_features, self.out_features, bias_attr=bias_attr
        )

        self.bias = self.linear.bias
        self.weight = self.linear.weight
        self.float_weight = self.linear.weight
        self.weight_scale = None
        # check weight quantize
        # self.weightQuantizeCPUGPUConsistenceCheck(self.float_weight)

        self.weight, self.weight_scale = Q.weight_quantize(
            (
                self.float_weight.to("musa:0")
                if self.weight_dtype == "int4"
                else self.weight.to("musa:0")
            ),
            algo=(
                "weight_only_int8"
                if self.weight_dtype == "int8"
                else "weight_only_int4"
            ),
            group_size=self.group_size,
        )

    def get_linear_out(self):
        out = self.linear(self.x)
        return out.numpy()

    def get_weight_only_linear_out(self):
        out = Q.weight_only_linear(
            self.x,
            self.weight,
            bias=self.bias,
            weight_scale=self.weight_scale,
            weight_dtype=self.weight_dtype,
            group_size=self.group_size,
        )
        return out.numpy()

    def test_weight_only_linear(self):
        out_expect = self.get_linear_out()
        out_real = self.get_weight_only_linear_out()

        if self.dtype == "bfloat16":
            out_real = convert_uint16_to_float(out_real)
            out_expect = convert_uint16_to_float(out_expect)
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )



class WeightOnlyLinearTestCase1(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"



class WeightOnlyLinearTestCase2(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int4"


class WeightOnlyLinearTestCase7(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.batch = 1
        self.token = 1



class WeightOnlyLinearTestCase8(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1


class WeightOnlyLinearTestCase17(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 64


class WeightOnlyLinearTestCase18(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.bias = False
        self.batch = 1
        self.token = 1
        self.group_size = 128


class WeightOnlyLinearTestCase22(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.in_features = 128
        self.out_features = 288



class WeightOnlyLinearTestCase23(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.bias = False
        self.weight_dtype = "int4"
        self.in_features = 128
        self.out_features = 288


class WeightOnlyLinearTestCase27(WeightOnlyLinearTestCase):
    def config(self):
        super().config()
        self.dtype = 'float16'
        self.weight_dtype = "int4"
        self.group_size = 128


if __name__ == '__main__':
    unittest.main()
