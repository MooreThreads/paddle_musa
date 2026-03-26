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

#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T, typename Context>
PADDLE_API void MaxRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& dims,
                             bool keep_dim,
                             bool reduce_all,
                             DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MaxFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(max_raw,
                   musa,
                   ALL_LAYOUT,
                   phi::MaxRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2) {}
