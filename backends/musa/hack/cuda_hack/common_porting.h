/* Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.

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
#include <type_traits>

// template<typename T>
// bool is_exists() {
//     if (std::is_same_v<T, void> == false) {
//         return true;
//     }
//     return false;
// }
// #define DEF(__C__, __M__) \
//   do { \
//    if (is_exists<cuda##__X__>() == false) { \
//     using __C__ __M__; \
//    } \
//   } while(false);


#define cudaEvent_t musaEvent_t
#define cudaStream_t musaStream_t
#define cudaEventCreate musaEventCreate
#define cudaEventRecord musaEventRecord
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaGetDevice musaGetDevice
#define cudaEventElapsedTime musaEventElapsedTime
#define cudaDataType_t musaDataType_t
#define cudaDataType musaDataType
#define cudaMemset musaMemset
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define __nv_bfloat16 __mt_bfloat16
