// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
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

#pragma once

#include <mutex>  // NOLINT

#include "musa_helper.h"

#include "paddle/fluid/platform/dynload/mublas.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

class CublasHandleHolder {
 public:
  explicit CublasHandleHolder(musaStream_t stream) {
    PADDLE_RETRY_CUDA_SUCCESS(dynload::mublasCreate(&handle_));
    PADDLE_RETRY_CUDA_SUCCESS(dynload::mublasSetStream(handle_, stream));
  }

  const mublasHandle_t& GetCublasHandle() const { return handle_; }

  ~CublasHandleHolder() PADDLE_MAY_THROW {
    PADDLE_RETRY_CUDA_SUCCESS(dynload::mublasDestroy(handle_));
  }

  template <typename Callback>
  inline void Call(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(mtx_);
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  mublasHandle_t handle_;
  mutable std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
