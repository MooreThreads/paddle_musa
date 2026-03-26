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


#include <cmath>
#include <string>

#include "paddle/phi/kernels/isfinite_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/kernels/funcs/isfinite_functor.h"

// check if vanilla float/double
template <typename T>
struct is_float_or_double
    : std::integral_constant<bool,
                             std::is_same<T, float>::value ||
                                 std::is_same<T, double>::value> {};

// check ifspecial float type, e.g. float16/bfloat16
template <typename T>
struct is_other_float
    : std::integral_constant<bool,
                             std::is_floating_point<T>::value &&
                                 !is_float_or_double<T>::value> {};

// check if complex type
template <typename T>
struct is_complex64_or_complex128
    : std::integral_constant<bool,
                             std::is_same<T, phi::complex64>::value ||
                                 std::is_same<T, phi::complex128>::value> {};

namespace phi {
using Tensor = DenseTensor;

/*
Codes for isfinite/isinf/isnan as constructed as below:
1. A general template,
2. partial specialization for regular floating-point numbers(float/double),
3. partial specialization for special floating-point numbers(float16/bfloat16
and other special float),
4. partial specialization for non-floating-point (integer) types.
5. partial specialization for complex types.
*/

/* IsfiniteFunctor */
template <typename DeviceContext, typename T, typename Enable = void>
struct IsfiniteFunctor {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename T>
struct IsfiniteFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      out_data[i] = true;
    }
  }
};

template <typename T>
struct IsfiniteFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_float_or_double<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isfinite(a);
    }
  }
};

template <typename T>
struct IsfiniteFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_other_float<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = phi::dtype::isfinite(a);
    }
  }
};

template <typename T>
struct IsfiniteFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isfinite(a.real) && std::isfinite(a.imag);
    }
  }
};

/* IsnanFunctor */
template <typename DeviceContext, typename T, typename Enable = void>
struct IsnanFunctor {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename T>
struct IsnanFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      out_data[i] = false;
    }
  }
};

template <typename T>
struct IsnanFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_float_or_double<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isnan(a);
    }
  }
};

template <typename T>
struct IsnanFunctor<phi::CPUContext,
                    T,
                    typename std::enable_if<is_other_float<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = phi::dtype::isnan(a);
    }
  }
};

template <typename T>
struct IsnanFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isnan(a.real) || std::isnan(a.imag);
    }
  }
};

/* IsinfFunctor */
template <typename DeviceContext, typename T, typename Enable = void>
struct IsinfFunctor {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename T>
struct IsinfFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    auto num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      out_data[i] = false;
    }
  }
};

template <typename T>
struct IsinfFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_float_or_double<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isinf(a);
    }
  }
};

template <typename T>
struct IsinfFunctor<phi::CPUContext,
                    T,
                    typename std::enable_if<is_other_float<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = phi::dtype::isinf(a);
    }
  }
};

template <typename T>
struct IsinfFunctor<
    phi::CPUContext,
    T,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t num = in.numel();
    for (int64_t i = 0; i < num; i++) {
      const T& a = in_a[i];
      out_data[i] = std::isinf(a.real) || std::isinf(a.imag);
    }
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
/* IsfiniteFunctor */
template <typename T, typename IndexType>
__global__ void IsfiniteCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isfinite(a);
  }
}

template <typename T, typename IndexType>
__global__ void IsfiniteCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    out_data[i] = true;
  }
}

template <typename T, typename IndexType>
__global__ void IsfiniteCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isfinite(a.real) && isfinite(a.imag);
  }
}

/* IsnanFunctor */
template <typename T, typename IndexType>
__global__ void IsnanCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isnan(a);
  }
}

template <typename T, typename IndexType>
__global__ void IsnanCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    out_data[i] = false;
  }
}

template <typename T, typename IndexType>
__global__ void IsnanCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isnan(a.real) || isnan(a.imag);
  }
}

/* IsinfFunctor */
template <typename T, typename IndexType>
__global__ void IsinfCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isinf(a);
  }
}

template <typename T, typename IndexType>
__global__ void IsinfCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    out_data[i] = false;
  }
}

template <typename T, typename IndexType>
__global__ void IsinfCUDAKernel(
    const T* in_data,
    IndexType num,
    bool* out_data,
    typename std::enable_if<is_complex64_or_complex128<T>::value>::type* = 0) {
  IndexType idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (IndexType i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = in_data[i];
    out_data[i] = isinf(a.real) || isinf(a.imag);
  }
}

template <typename T>
struct IsfiniteFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int64_t num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t block = 1024;
    int64_t grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
    IsfiniteCUDAKernel<T, int64_t>
          <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};

template <typename T>
struct IsnanFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int64_t num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t block = 1024;
    int64_t grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
    IsnanCUDAKernel<T, int64_t>
          <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};

template <typename T>
struct IsinfFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int64_t num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int64_t block = 1024;
    int64_t grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
    IsinfCUDAKernel<T, int64_t>
          <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};
#endif

template <typename T, typename Context>
PADDLE_API void IsfiniteKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  IsfiniteFunctor<Context, T>()(dev_ctx, x, out);
}
template <typename T, typename Context>
void IsinfKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  IsinfFunctor<Context, T>()(dev_ctx, x, out);
}
template <typename T, typename Context>
void IsnanKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  IsnanFunctor<Context, T>()(dev_ctx, x, out);
}
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(isinf,
                   musa,
                   ALL_LAYOUT,
                   phi::IsinfKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_CUSTOM_KERNEL_REGISTER(isnan,
                   musa,
                   ALL_LAYOUT,
                   phi::IsnanKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_CUSTOM_KERNEL_REGISTER(isfinite,
                   musa,
                   ALL_LAYOUT,
                   phi::IsfiniteKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
