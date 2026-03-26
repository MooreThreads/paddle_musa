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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/legacy/elementwise_add_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_subtract_kernel.h"


namespace phi {

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::AddRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  phi::AddRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::MultiplyRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::DivideRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
PADDLE_API void SubtractKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::SubtractRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  int axis = -1;
  RemainderRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void TruncDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::TruncDivideFunctor<T>(), axis);
}

// Create the definition of Heaviside
template <typename T, typename Context>
void HeavisideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::ElementwiseHeavisideFunctor<T>());
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::CopySignFunctor<T>());
}

template <typename T, typename Context>
void NextafterKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::NextafterFunctor<T>());
}

} // namespace phi

PD_CUSTOM_KERNEL_REGISTER(maximum,
                   musa,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(minimum,
                   musa,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(remainder,
                   musa,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::complex64,
                   phi::complex128,
                   phi::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(floor_divide,
                   musa,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(trunc_divide,
                   musa,
                   ALL_LAYOUT,
                   phi::TruncDivideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(elementwise_pow,
                   musa,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(copysign,
                   musa,
                   ALL_LAYOUT,
                   phi::CopySignKernel,
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
PD_CUSTOM_KERNEL_REGISTER(
    nextafter, musa, ALL_LAYOUT, phi::NextafterKernel, float, double) {}

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = phi::complex64;
using complex128 = phi::complex128;

PD_CUSTOM_KERNEL_REGISTER(fmax,
                   musa,
                   ALL_LAYOUT,
                   phi::FMaxKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin,
                   musa,
                   ALL_LAYOUT,
                   phi::FMinKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside,
                   musa,
                   ALL_LAYOUT,
                   phi::HeavisideKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(add,
                musa,
                ALL_LAYOUT,
                phi::AddKernel,
                float,
                double,
                int16_t,
                int,
                bool,
                uint8_t,
                int8_t,
                int64_t,
                float16,
                bfloat16,
                complex64,
                complex128) {}
 
PD_CUSTOM_KERNEL_REGISTER(grad_add,
                   musa,
                   ALL_LAYOUT,
                   phi::GradAddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_CUSTOM_KERNEL_REGISTER(divide,
                   musa,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_CUSTOM_KERNEL_REGISTER(multiply,
                   musa,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128,
                   bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(subtract,
                   musa,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}
