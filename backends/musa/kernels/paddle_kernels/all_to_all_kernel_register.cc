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

#ifdef STREAM_TYPE
#undef STREAM_TYPE
#define STREAM_TYPE void*
#endif

#include "paddle/phi/kernels/all_to_all_kernel.h"
#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_MCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/distributed/utils.h"
#endif

namespace phi {

template <typename T, typename Context>
void AllToAllKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  auto x_dims = x.dims();
  out->Resize(x_dims);
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("XCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  auto stream = dev_ctx.stream();
  PADDLE_ENFORCE_NOT_NULL(stream,
                          errors::NotFound("Should initialize XCCL firstly."));

  int nranks = comm_ctx->GetSize();
  int send_numel = x.numel() / nranks;
  size_t offset = 0;

  PADDLE_ENFORCE_EQ(
      x_dims[0] % nranks,
      0,
      errors::InvalidArgument(
          "The first dimension size (%d) of the input tensor must be "
          "divisible by the number of ranks (%d).",
          x_dims[0],
          nranks));

  comm_ctx->GroupStart();

  const auto* send_buf = x.data<T>();
  auto* recv_buf = out->data<T>();
  for (auto i = 0; i < nranks; ++i) {
    auto send_buf = phi::distributed::GetPartialTensor(x, offset, send_numel);
    comm_ctx->Send(send_buf, send_numel, i, stream);
    auto recv_buf =
        phi::distributed::GetPartialTensor(*out, offset, send_numel);
    comm_ctx->Recv(&recv_buf, send_numel, i, stream);
    offset += send_numel;
  }
  comm_ctx->GroupEnd();
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(all_to_all,
                   musa,
                   ALL_LAYOUT,
                   phi::AllToAllKernel,
                   float,
                   double,
                   int,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

#undef STREAM_TYPE
#define STREAM_TYPE musaStream_t