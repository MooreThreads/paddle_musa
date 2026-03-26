// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_MUSA)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#else  // PADDLE_WITH_CUDA
#ifndef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#else
#include <musa_runtime.h>
#endif
#endif

namespace phi {

// Note(qili93): CUDA Runtime API supported by HIP
// https://github.com/ROCm/HIPIFY/blob/master/doc/markdown/CUDA_Runtime_API_functions_supported_by_HIP.md

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = ROCM_TYPE;
//#elif defined(PADDLE_WITH_MUSA)
//#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
//  using GPU_TYPE = MUSA_TYPE;
#else  // PADDLE_WITH_CUDA
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif

DECLARE_TYPE_FOR_GPU(gpuError_t, musaError_t, hipError_t);
DECLARE_TYPE_FOR_GPU(gpuMemcpyKind, musaMemcpyKind, hipMemcpyKind);
DECLARE_TYPE_FOR_GPU(gpuDeviceProp, musaDeviceProp, hipDeviceProp_t);
#ifndef PADDLE_WITH_CUSTOM_DEVICE
DECLARE_TYPE_FOR_GPU(dnnDataType_t, cudnnDataType_t, miopenDataType_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingMode_t, cudnnPoolingMode_t, miopenPoolingMode_t);
DECLARE_TYPE_FOR_GPU(dnnTensorFormat_t,
                     cudnnTensorFormat_t,
                     miopenTensorFormat_t);
DECLARE_TYPE_FOR_GPU(dnnActivationMode_t,
                     cudnnActivationMode_t,
                     miopenActivationMode_t);
#endif
DECLARE_TYPE_FOR_GPU(gpuGraph_t, musaGraph_t, hipGraph_t);
DECLARE_TYPE_FOR_GPU(gpuFunction_t, musaFunction_t, hipFunction_t);
DECLARE_TYPE_FOR_GPU(gpuGraphExec_t, musaGraphExec_t, hipGraphExec_t);
DECLARE_TYPE_FOR_GPU(gpuGraphNode_t, musaGraphNode_t, hipGraphNode_t);
DECLARE_TYPE_FOR_GPU(gpuGraphNodeType, musaGraphNodeType, hipGraphNodeType);
DECLARE_TYPE_FOR_GPU(gpuKernelNodeParams,
                     musaKernelNodeParams,
                     hipKernelNodeParams);
DECLARE_TYPE_FOR_GPU(gpuStreamCaptureMode,
                     musaStreamCaptureMode,
                     hipStreamCaptureMode);
DECLARE_TYPE_FOR_GPU(gpuStreamCaptureStatus,
                     musaStreamCaptureStatus,
                     hipStreamCaptureStatus);

#undef DECLARE_TYPE_FOR_GPU

#ifdef PADDLE_WITH_HIP
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = ROCM_CV;
//#elif defined(PADDLE_WITH_MUSA)
//#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
//  constexpr auto GPU_CV = MUSA_CV;
#else  // PADDLE_WITH_CUDA
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = CUDA_CV;
#endif

DECLARE_CONSTANT_FOR_GPU(gpuErrorOutOfMemory,
                         musaErrorMemoryAllocation,
                         hipErrorOutOfMemory);
DECLARE_CONSTANT_FOR_GPU(gpuErrorNotReady, musaErrorNotReady, hipErrorNotReady);
DECLARE_CONSTANT_FOR_GPU(gpuSuccess, musaSuccess, hipSuccess);

DECLARE_CONSTANT_FOR_GPU(gpuMemcpyHostToDevice,
                         musaMemcpyKind::musaMemcpyHostToDevice,
                         hipMemcpyKind::hipMemcpyHostToDevice);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToHost,
                         musaMemcpyKind::musaMemcpyDeviceToHost,
                         hipMemcpyKind::hipMemcpyDeviceToHost);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToDevice,
                         musaMemcpyKind::musaMemcpyDeviceToDevice,
                         hipMemcpyKind::hipMemcpyDeviceToDevice);
DECLARE_CONSTANT_FOR_GPU(gpuEventDisableTiming,
                         musaEventDisableTiming,
                         hipEventDisableTiming);
DECLARE_CONSTANT_FOR_GPU(gpuStreamNonBlocking,
                         musaStreamNonBlocking,
                         hipStreamNonBlocking);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureModeThreadLocal,
                         musaStreamCaptureModeThreadLocal,
                         hipStreamCaptureModeThreadLocal);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureModeRelaxed,
                         musaStreamCaptureModeRelaxed,
                         hipStreamCaptureModeRelaxed);
DECLARE_CONSTANT_FOR_GPU(gpuStreamCaptureStatusActive,
                         musaStreamCaptureStatusActive,
                         hipStreamCaptureStatusActive);
DECLARE_CONSTANT_FOR_GPU(gpuGraphNodeTypeKernel,
                         musaGraphNodeTypeKernel,
                         hipGraphNodeTypeKernel);

#undef DECLARE_CONSTANT_FOR_GPU

#ifdef PADDLE_WITH_HIP
#define DECLARE_FUNCTION_FOR_GPU(GPU_FUNC, CUDA_FUNC, ROCM_FUNC) \
  const auto GPU_FUNC = ROCM_FUNC;
#else  // PADDLE_WITH_CUDA
#define DECLARE_FUNCTION_FOR_GPU(GPU_FUNC, CUDA_FUNC, ROCM_FUNC)                          \
  const auto GPU_FUNC = CUDA_FUNC;
#endif

inline musaError_t musaEventCreateWithFlagsHost(musaEvent_t *e, unsigned int flags) {
    return musaEventCreateWithFlags(e, flags);
}

inline musaError_t musaEventRecordHost(musaEvent_t e, musaStream_t s) {
    return musaEventRecord(e, s);
}

inline musaError_t musaEventDestroyHost(musaEvent_t e) {
    return musaEventDestroy(e);
}

DECLARE_FUNCTION_FOR_GPU(gpuGraphGetNodes, musaGraphGetNodes, hipGraphGetNodes);
DECLARE_FUNCTION_FOR_GPU(gpuGraphGetEdges, musaGraphGetEdges, hipGraphGetEdges);
DECLARE_FUNCTION_FOR_GPU(gpuGraphLaunch, musaGraphLaunch, hipGraphLaunch);
DECLARE_FUNCTION_FOR_GPU(gpuGraphDestroy, musaGraphDestroy, hipGraphDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuGraphExecDestroy,
                         musaGraphExecDestroy,
                         hipGraphExecDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuGraphNodeGetType,
                         musaGraphNodeGetType,
                         hipGraphNodeGetType);
DECLARE_FUNCTION_FOR_GPU(gpuGraphExecKernelNodeSetParams,
                         musaGraphExecKernelNodeSetParams,
                         hipGraphExecKernelNodeSetParams);
DECLARE_FUNCTION_FOR_GPU(gpuGraphKernelNodeGetParams,
                         musaGraphKernelNodeGetParams,
                         hipGraphKernelNodeGetParams);
DECLARE_FUNCTION_FOR_GPU(gpuStreamCreateWithPriority,
                         musaStreamCreateWithPriority,
                         hipStreamCreateWithPriority);
DECLARE_FUNCTION_FOR_GPU(gpuStreamBeginCapture,
                         musaStreamBeginCapture,
                         hipStreamBeginCapture);
DECLARE_FUNCTION_FOR_GPU(gpuStreamEndCapture,
                         musaStreamEndCapture,
                         hipStreamEndCapture);
DECLARE_FUNCTION_FOR_GPU(gpuStreamGetCaptureInfo,
                         musaStreamGetCaptureInfo,
                         hipStreamGetCaptureInfo);
DECLARE_FUNCTION_FOR_GPU(gpuEventCreateWithFlags,
                         musaEventCreateWithFlagsHost,
                         hipEventCreateWithFlags);
DECLARE_FUNCTION_FOR_GPU(gpuEventRecord, musaEventRecordHost, hipEventRecord);
DECLARE_FUNCTION_FOR_GPU(gpuEventDestroy, musaEventDestroyHost, hipEventDestroy);
DECLARE_FUNCTION_FOR_GPU(gpuEventQuery, musaEventQuery, hipEventQuery);
DECLARE_FUNCTION_FOR_GPU(gpuEventSynchronize,
                         musaEventSynchronize,
                         hipEventSynchronize);

#undef DECLARE_FUNCTION_FOR_GPU

}  // namespace phi

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_MUSA)
