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


#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reduce_kernel_impl.h"
#include "paddle/phi/backends/custom/custom_context.h"


namespace phi {

using XIntArray = paddle::experimental::IntArrayBase<phi::DenseTensor>;

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const XIntArray& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out) {
  const bool reduce_all = recompute_reduce_all(x, dims);
  SumRawKernel<T, Context>(dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

#define INSTANTIATE_SUM(T)                                                     \
  template void SumKernel<T, phi::CustomContext>(                              \
      const phi::CustomContext&,                                               \
      const phi::DenseTensor&,                                                 \
      const paddle::experimental::IntArrayBase<phi::DenseTensor>&,             \
      phi::DataType,                                                           \
      bool,                                                                    \
      phi::DenseTensor*);

INSTANTIATE_SUM(bool)
INSTANTIATE_SUM(float)
INSTANTIATE_SUM(double)
INSTANTIATE_SUM(phi::float16)
INSTANTIATE_SUM(phi::bfloat16)
INSTANTIATE_SUM(int16_t)
INSTANTIATE_SUM(int)
INSTANTIATE_SUM(int64_t)
INSTANTIATE_SUM(uint8_t)
INSTANTIATE_SUM(int8_t)
INSTANTIATE_SUM(phi::complex64)
INSTANTIATE_SUM(phi::complex128)

#undef INSTANTIATE_SUM
}


PD_CUSTOM_KERNEL_REGISTER(sum,
                   musa,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}