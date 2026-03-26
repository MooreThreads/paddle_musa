// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"

namespace phi {

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  // skip out
  auto* out = &dout;
  if (dout.numel() == 0) {
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      if (dx->numel() != 0) {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
      }
    }
    if (dy) {
      dev_ctx.template Alloc<T>(dy);
      if (dy->numel() != 0) {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dy->dims())), 0, dy);
      }
    }
    return;
  }
  if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
    elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy);
  } else {
    default_elementwise_sub_grad<T>(dev_ctx, x, y, *out, dout, dx, dy, axis);
  }
}

template <typename T, typename Context>
void SubtractDoubleGradImpl(const Context& dev_ctx,
                            const DenseTensor& y,
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
                            const DenseTensor& dout,
                            int axis,
                            DenseTensor* ddout) {
  // DDOut = ddx - ddy
  if (ddout) {
    DenseTensor ddx_safe, ddy_safe;
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, dout, ddx.get_ptr(), &ddx_safe);
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, y, ddy.get_ptr(), &ddy_safe);

    dev_ctx.template Alloc<T>(ddout);
    funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
        dev_ctx, ddx_safe, ddy_safe, funcs::SubtractFunctor<T>(), ddout, axis);
  }
}

template <typename T, typename Context>
void SubtractDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              int axis,
                              DenseTensor* ddout) {
  phi::SubtractDoubleGradImpl<T>(dev_ctx, y, ddx, ddy, dout, axis, ddout);
}

}


PD_CUSTOM_KERNEL_REGISTER(fmax_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::ElementwiseFMaxGradKernel,
                   float,
                   double,
                   int,
                   phi::float16,
                   phi::bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::ElementwiseFMinGradKernel,
                   float,
                   double,
                   int,
                   phi::float16,
                   phi::bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(maximum_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(minimum_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(remainder_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::RemainderGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::HeavisideGradKernel,
                   float,
                   double,
                   int,
                   phi::float16,
                   phi::bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(elementwise_pow_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::ElementwisePowGradKernel,
                   float,
                   double,
                   int,
                   phi::float16,
                   phi::bfloat16,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(add_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::AddGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(add_double_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::AddDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(add_triple_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::AddTripleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(divide_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::DivideGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(divide_double_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::DivideDoubleGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::MultiplyGradKernel,
                   float,
                   phi::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_double_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::MultiplyDoubleGradKernel,
                   float,
                   phi::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_triple_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::MultiplyTripleGradKernel,
                   float,
                   phi::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(subtract_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::SubtractGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(subtract_double_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::SubtractDoubleGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(copysign_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::CopySignGradKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
