/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/softmax_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

#include <musa_runtime.h>

using ::musa::dnn::Softmax;
using ::musa::dnn::MemoryHandler;
using muTensor = ::musa::dnn::Tensor;

namespace phi {

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int axis,
                       DenseTensor* x_grad) {
    auto& h = GetMudnnHandle<Context>(dev_ctx);
    ::musa::dnn::Softmax ddnSoftmax;
    CHECK_MUDNN_STATUS(ddnSoftmax.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE), "SetAlgorithm");
    CHECK_MUDNN_STATUS(ddnSoftmax.SetMode(::musa::dnn::Softmax::Mode::SOFTMAX), "SetMode");
    CHECK_MUDNN_STATUS(ddnSoftmax.SetDim(axis), "SetDim");


    dev_ctx.template Alloc<T>(x_grad);
    auto musa_out = CreateMUTensor(out);
    auto musa_out_grad = CreateMUTensor(out_grad);
    auto musa_x_grad = CreateMUTensor(*x_grad);

    auto place = dev_ctx.GetPlace();
    ::musa::dnn::MemoryMaintainer maintainer =
        [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

    CHECK_MUDNN_STATUS(
        ddnSoftmax.RunBwd(
            h,
            musa_x_grad,
            musa_out,
            musa_out_grad,
            maintainer),
        "Run Mudnn Softmax Bwd.");
}

}

PD_CUSTOM_KERNEL_REGISTER(softmax_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::SoftmaxGradKernel,
                          float,
                          phi::float16,
                          phi::bfloat16) {}
