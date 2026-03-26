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


paddle.enable_static()
paddle.set_device("musa")


class TestAdamRowOp(unittest.TestCase):
    def test_adamw_row_op_forward_all_dtypes(self):
        B, T = 32, 10
        V, D = 1000, 64

        ids = paddle.static.data(name="ids", shape=[B, T], dtype="int64")
        emb = paddle.nn.Embedding(num_embeddings=V, embedding_dim=D, sparse=True)
        loss = paddle.sum(emb(ids))

        opt = paddle.optimizer.AdamW(learning_rate=1e-3)
        opt.minimize(loss)

        startup = paddle.static.default_startup_program()
        main = paddle.static.default_main_program()

        init_weight = np.random.randn(V, D).astype("float32")
        feed_ids = np.random.randint(0, V, size=(B, T), dtype="int64")
        param_name = emb.weight.name

        cpu_scope = paddle.static.Scope()
        with paddle.static.scope_guard(cpu_scope):
            cpu_exe = paddle.static.Executor(paddle.CPUPlace())
            cpu_exe.run(startup)
            cpu_tensor = cpu_scope.find_var(param_name).get_tensor()
            cpu_tensor.set(init_weight, paddle.CPUPlace())
            cpu_exe.run(main, feed={"ids": feed_ids})
            cpu_weight = np.array(cpu_scope.find_var(param_name).get_tensor())

        musa_scope = paddle.static.Scope()
        with paddle.static.scope_guard(musa_scope):
            musa_exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
            musa_exe.run(startup)
            musa_tensor = musa_scope.find_var(param_name).get_tensor()
            musa_tensor.set(init_weight, paddle.CustomPlace("musa", 0))
            musa_exe.run(main, feed={"ids": feed_ids})
            musa_weight = np.array(musa_scope.find_var(param_name).get_tensor())

        self.assertTrue(np.allclose(cpu_weight, musa_weight, rtol=1e-6, atol=1e-6))
