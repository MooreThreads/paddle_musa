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

#include "paddle/phi/kernels/p_recv_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/send_recv_functor.h"

#if defined(PADDLE_WITH_MCCL)
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PRecvKernel(const Context& dev_ctx,
                 int peer,
                 DataType dtype,
                 const std::vector<int>& out_shape,
                 bool dynamic_shape,
                 DenseTensor* out) {
  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  auto stream = dev_ctx.stream();

  // auto data_type = phi::TransToPhiDataType(dtype);
  if (dynamic_shape) {
    DDim new_dim =
        recv_shape_info<Context, distributed::XCCLCommContext, phi::stream::stream_t>(
            dev_ctx, out, comm_ctx, peer);
    out->Resize(new_dim);
  }
  dev_ctx.Alloc(out, dtype);
  comm_ctx->Recv(out, out->numel(), peer, stream);
}

template <typename T, typename Context>
void PRecvArrayKernel(const Context& dev_ctx,
                      int peer,
                      DataType dtype,
                      const std::vector<int>& out_shape,
                      TensorArray* out_array) {
  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  auto stream = dev_ctx.stream();
  for (size_t idx = 0; idx < out_shape.size(); ++idx) {
    VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
    auto out = out_array->at(idx);
    auto out_dims = out.dims();
    dev_ctx.Alloc(&out, dtype);
    comm_ctx->Recv(&out, out.numel(), peer, stream);
    VLOG(3) << "rank " << comm_ctx->GetRank() << " recv "
            << common::product(out_dims) << " from " << peer;
  }
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(p_recv,
                   musa,
                   ALL_LAYOUT,
                   phi::PRecvKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int64_t,
                   phi::bfloat16,
                   phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(p_recv_array,
                   musa,
                   ALL_LAYOUT,
                   phi::PRecvArrayKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::bfloat16,
                   phi::float16) {}

#undef STREAM_TYPE
#define STREAM_TYPE musaStream_t