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

#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/complex.h"

namespace phi {

template <typename T, typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const paddle::experimental::IntArrayBase<DenseTensor>& shape,
                 DataType dtype,
                 DenseTensor* out) {
  DataType target = (dtype == DataType::UNDEFINED)
                      ? CppTypeToDataType<T>::Type()
                      : dtype;
  PADDLE_ENFORCE_EQ(
      target, CppTypeToDataType<T>::Type(),
      phi::errors::InvalidArgument("EmptyKernel dtype (%s) mismatches T (%s).",
                                   DataTypeToString(target),
                                   DataTypeToString(CppTypeToDataType<T>::Type())));
  out->Resize(make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
}

template <typename T, typename Context>
void EmptyLikeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DataType dtype,
                     DenseTensor* out) {
  DataType target = (dtype == DataType::UNDEFINED) ? x.dtype() : dtype;
  PADDLE_ENFORCE_EQ(
      target, CppTypeToDataType<T>::Type(),
      phi::errors::InvalidArgument("EmptyLikeKernel dtype (%s) mismatches T (%s).",
                                   DataTypeToString(target),
                                   DataTypeToString(CppTypeToDataType<T>::Type())));
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(empty,
                   musa,
                   ALL_LAYOUT,
                   phi::EmptyKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(empty_like,
                   musa,
                   ALL_LAYOUT,
                   phi::EmptyLikeKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}