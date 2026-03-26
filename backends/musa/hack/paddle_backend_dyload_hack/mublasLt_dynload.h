/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mublasLt.h>
#include <cuda.h>

#include <mutex>  // NOLINT
#include <type_traits>
#include "common_porting.h"

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag mublasLt_dso_flag;
extern void *mublasLt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mublasLt routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
// #define CONCAT_INNER(__prefix, __name) __prefix##__name 
// #define CONCAT(__prefix, __name) CONCAT_INNER(__prefix, __name)
// 
// #define STRINGIFY_INNER(x) #x        
// #define STRINGIFY(x) STRINGIFY_INNER(x)

#define DECLARE_DYNAMIC_LOAD_MUBLASLT_WRAP(__name)                          \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using mublasLt_func =                                                 \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(mublasLt_dso_flag, []() {                              \
        mublasLt_dso_handle = phi::dynload::GetMublasLtDsoHandle();         \
      });                                                                   \
      static void *p_##__name = dlsym(mublasLt_dso_handle, #__name);        \
      return reinterpret_cast<mublasLt_func>(p_##__name)(args...);          \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name


#if 1
#define MUBLASLT_BLAS_ROUTINE_EACH(__macro)         \
  __macro(mublasLtCreate);                          \
  __macro(mublasLtDestroy);                         \
  __macro(mublasLtMatmul);                          \
  __macro(mublasLtMatmulDescCreate);                \
  __macro(mublasLtMatmulDescDestroy);               \
  __macro(mublasLtMatmulDescSetAttribute);          \
  __macro(mublasLtMatmulDescGetAttribute);          \
  __macro(mublasLtMatrixLayoutCreate);              \
  __macro(mublasLtMatrixLayoutDestroy);             \
  __macro(mublasLtMatrixLayoutSetAttribute);        \
  __macro(mublasLtMatrixLayoutGetAttribute);        \
  __macro(mublasLtMatmulPreferenceCreate);          \
  __macro(mublasLtMatmulPreferenceDestroy);         \
  __macro(mublasLtMatmulPreferenceSetAttribute);    \
  __macro(mublasLtMatmulAlgoGetHeuristic);          \
  __macro(mublasLtMatrixTransform);                 \
  __macro(mublasLtMatrixTransformDescCreate);       \
  __macro(mublasLtMatrixTransformDescDestroy);      \
  __macro(mublasLtMatrixTransformDescSetAttribute); \
  __macro(mublasLtMatmulAlgoInit);                  \
  __macro(mublasLtMatmulAlgoConfigSetAttribute);    \
  __macro(mublasLtMatmulAlgoConfigGetAttribute);    \
  __macro(mublasLtMatmulAlgoGetIds);                \
  __macro(mublasLtMatmulAlgoCapGetAttribute);       \
  __macro(mublasLtMatmulAlgoCheck);                 \
  __macro(mublasLtGetCudartVersion);
#else
#define MUBLASLT_BLAS_ROUTINE_EACH(__macro)      \
  __macro(mublasLtCreate);                       \
  __macro(mublasLtDestroy);                      \
  __macro(mublasLtMatmul);                       \
  __macro(mublasLtMatmulDescCreate);             \
  __macro(mublasLtMatmulDescDestroy);            \
  __macro(mublasLtMatmulDescSetAttribute);       \
  __macro(mublasLtMatmulDescGetAttribute);       \
  __macro(mublasLtMatrixLayoutCreate);           \
  __macro(mublasLtMatrixLayoutDestroy);          \
  __macro(mublasLtMatrixLayoutSetAttribute);     \
  __macro(mublasLtMatrixLayoutGetAttribute);     \
  __macro(mublasLtMatmulPreferenceCreate);       \
  __macro(mublasLtMatmulPreferenceDestroy);      \
  __macro(mublasLtMatmulPreferenceSetAttribute); \
  __macro(mublasLtMatmulAlgoGetHeuristic);       \
  __macro(mublasLtMatrixTransform);              \
  __macro(mublasLtMatrixTransformDescCreate);    \
  __macro(mublasLtMatrixTransformDescDestroy);   \
  __macro(mublasLtMatrixTransformDescSetAttribute);
#endif

MUBLASLT_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUBLASLT_WRAP)
// #endif

#undef DECLARE_DYNAMIC_LOAD_MUBLASLT_WRAP

}  // namespace dynload
}  // namespace phi

#define cublasLtHandle_t mublasLtHandle_t
#define cublasLtMatmulAlgo_t mublasLtMatmulAlgo_t
#define cublasLtMatmulDesc_t mublasLtMatmulDesc_t
#define cublasLtMatrixLayout_t mublasLtMatrixLayout_t
#define cublasLtMatrixLayout_t mublasLtMatrixLayout_t
#define cublasComputeType_t mublasComputeType_t
#define cudaDataType_t musaDataType_t
#define cublasStatus_t mublasStatus_t
#define cublasLtMatmulHeuristicResult_t mublasLtMatmulHeuristicResult_t

#define cublasLtCreate mublasLtCreate
#define cublasLtDestroy mublasLtDestroy
#define cublasLtMatmul mublasLtMatmul
#define cublasLtMatmulDescCreate mublasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy mublasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute mublasLtMatmulDescSetAttribute
#define cublasLtMatmulDescGetAttribute mublasLtMatmulDescGetAttribute
#define cublasLtMatrixLayoutCreate mublasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy mublasLtMatrixLayoutDestroy
#define cublasLtMatrixLayoutSetAttribute mublasLtMatrixLayoutSetAttribute
#define cublasLtMatrixLayoutGetAttribute mublasLtMatrixLayoutGetAttribute
#define cublasLtMatmulPreferenceCreate mublasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy mublasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute mublasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulAlgoGetHeuristic mublasLtMatmulAlgoGetHeuristic
#define cublasLtMatrixTransform mublasLtMatrixTransform
#define cublasLtMatrixTransformDescCreate mublasLtMatrixTransformDescCreate
#define cublasLtMatrixTransformDescDestroy mublasLtMatrixTransformDescDestroy
#define cublasLtMatrixTransformDescSetAttribute mublasLtMatrixTransformDescSetAttribute
#define cublasLtMatmulAlgoInit mublasLtMatmulAlgoInit
#define cublasLtMatmulAlgoConfigSetAttribute mublasLtMatmulAlgoConfigSetAttribute
#define cublasLtMatmulAlgoConfigGetAttribute mublasLtMatmulAlgoConfigGetAttribute
#define cublasLtMatmulAlgoGetIds mublasLtMatmulAlgoGetIds
#define cublasLtMatmulAlgoCapGetAttribute mublasLtMatmulAlgoCapGetAttribute
#define cublasLtMatmulAlgoCheck mublasLtMatmulAlgoCheck
#define cublasLtGetCudartVersion mublasLtGetCudartVersion

#define cublasLtEpilogue_t mublasLtEpilogue_t
#define cublasLtMatmulPreference_t mublasLtMatmulPreference_t

#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#define CUBLASLT_ALGO_CONFIG_TILE_ID MUBLASLT_ALGO_CONFIG_TILE_ID
#define CUBLASLT_ALGO_CONFIG_SPLITK_NUM MUBLASLT_ALGO_CONFIG_SPLITK_NUM
#define CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING MUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
#define CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME MUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME
#define CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT MUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT
#define CUBLASLT_REDUCTION_SCHEME_MASK MUBLASLT_REDUCTION_SCHEME_MASK
#define CUBLASLT_REDUCTION_SCHEME_NONE MUBLASLT_REDUCTION_SCHEME_NONE
#define CUBLASLT_MATMUL_DESC_TRANSA MUBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB MUBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE MUBLASLT_MATMUL_DESC_EPILOGUE

#define CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT MUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT
#define CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX MUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX
#define CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK MUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK
#define CUBLASLT_ALGO_CAP_SPLITK_SUPPORT MUBLASLT_ALGO_CAP_SPLITK_SUPPORT
#define CUBLASLT_ALGO_CAP_STAGES_IDS MUBLASLT_ALGO_CAP_STAGES_IDS
#define CUBLASLT_ALGO_CAP_TILE_IDS MUBLASLT_ALGO_CAP_TILE_IDS
#define CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION MUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
#define CUBLASLT_ALGO_CONFIG_STAGES_ID MUBLASLT_ALGO_CONFIG_STAGES_ID
#define CUBLASLT_MATMUL_STAGES_UNDEFINED MUBLASLT_MATMUL_STAGES_UNDEFINED
#define CUBLASLT_MATMUL_TILE_UNDEFINED MUBLASLT_MATMUL_TILE_UNDEFINED
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT MUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATRIX_LAYOUT_COLS MUBLASLT_MATRIX_LAYOUT_COLS
#define CUBLASLT_MATRIX_LAYOUT_ROWS MUBLASLT_MATRIX_LAYOUT_ROWS
#define CUBLASLT_MATRIX_LAYOUT_TYPE MUBLASLT_MATRIX_LAYOUT_TYPE

#define CUBLASLT_MATMUL_DESC_BIAS_POINTER MUBLASLT_MATMUL_DESC_BIAS_POINTER
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER MUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD MUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_EPILOGUE_BGRADA MUBLASLT_EPILOGUE_BGRADA
#define CUBLASLT_EPILOGUE_BGRADB MUBLASLT_EPILOGUE_BGRADB
#define CUBLASLT_EPILOGUE_BIAS MUBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_DEFAULT MUBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_EPILOGUE_DGELU MUBLASLT_EPILOGUE_DGELU
#define CUBLASLT_EPILOGUE_DRELU MUBLASLT_EPILOGUE_DRELU
#define CUBLASLT_EPILOGUE_GELU_AUX_BIAS MUBLASLT_EPILOGUE_GELU_AUX_BIAS
#define CUBLASLT_EPILOGUE_GELU_BIAS MUBLASLT_EPILOGUE_GELU_BIAS
#define CUBLASLT_EPILOGUE_RELU MUBLASLT_EPILOGUE_RELU
#define CUBLASLT_EPILOGUE_RELU_AUX_BIAS MUBLASLT_EPILOGUE_RELU_AUX_BIAS
#define CUBLASLT_EPILOGUE_RELU_BIAS MUBLASLT_EPILOGUE_RELU_BIAS
#define CUBLASLT_MATRIX_LAYOUT_LD MUBLASLT_MATRIX_LAYOUT_LD
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#define CUBLAS_COMPUTE_32I MUBLAS_COMPUTE_32I
#define CUBLAS_COMPUTE_64F MUBLAS_COMPUTE_64F
