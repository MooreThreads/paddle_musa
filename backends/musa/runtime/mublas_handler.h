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
#include "mublasLt_dynload.h" // NOLINT
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"

#include "glog/logging.h"


namespace musa {
namespace blas {

C_Status InitBlasHandle(const C_Device device,
                        C_BLASHandle *blas_handle,
                        C_Stream stream) {
  VLOG(1) << "init blas here";
  MUSA_CHECK(musaSetDevice(device->id));
  phi::dynload::mublasCreate(
      reinterpret_cast<mublasHandle_t *>(blas_handle));
  phi::dynload::mublasSetStream(
      *reinterpret_cast<mublasHandle_t *>(blas_handle),
      reinterpret_cast<musaStream_t>((stream)));
  VLOG(1) << "init blas done " << *blas_handle;
  return C_SUCCESS;
}

C_Status InitBlasLtHandle(const C_Device device,
                          C_BLASLtHandle *blaslt_handle) {
  MUSA_CHECK(musaSetDevice(device->id));
  phi::dynload::mublasLtCreate(
      reinterpret_cast<mublasLtHandle_t *>(blaslt_handle));
  return C_SUCCESS;
}

C_Status DestroyBlasHandle(const C_Device device, C_BLASHandle blas_handle) {
  VLOG(1) << "destroy handle";
  MUSA_CHECK(musaSetDevice(device->id));
  if (blas_handle != nullptr) {
    phi::dynload::mublasDestroy(reinterpret_cast<mublasHandle_t>(blas_handle));
    blas_handle = nullptr;
  }
  VLOG(1) << "destroy handle done";
  return C_SUCCESS;
}

C_Status DestroyBlasLtHandle(const C_Device device,
                             C_BLASLtHandle blaslt_handle) {
  MUSA_CHECK(musaSetDevice(device->id));
  if (blaslt_handle != nullptr) {
    phi::dynload::mublasLtDestroy(
        reinterpret_cast<mublasLtHandle_t>(blaslt_handle));
    blaslt_handle = nullptr;
  }
  return C_SUCCESS;
}

C_Status BlasSetMathMode(const C_Device device,
                        C_BLASHandle blas_handle,
                        int math_mode) {
  MUSA_CHECK(musaSetDevice(device->id));
  // enum BLASMathMode {
  //   BLAS_DEFAULT_MATH = 0,
  //   BLAS_TENSOR_OP_MATH = 1,
  //   BLAS_TF32_TENSOR_OP_MATH = 2
  // };
  mublasMath_t mode = static_cast<mublasMath_t>(math_mode);
  if (math_mode == 1) {
    mode = MUBLAS_TENSOR_OP_MATH;
  }
  phi::dynload::mublasSetMathMode(
    reinterpret_cast<mublasHandle_t>(blas_handle), static_cast<mublasMath_t>(math_mode));
  VLOG(1) << "set math mode done " << static_cast<mublasMath_t>(math_mode) << " " << blas_handle;
  return C_SUCCESS;
}

} // namespace blas
} // namespace musa
