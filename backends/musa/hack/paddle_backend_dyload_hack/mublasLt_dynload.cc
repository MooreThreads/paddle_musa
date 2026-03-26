/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
rights reserved.
- [Adapt blasLt dynload to musa backend]
*/

#include "mublasLt_dynload.h" // NOLINT

namespace phi::dynload {
std::once_flag mublasLt_dso_flag;
void *mublasLt_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MUBLASLT_BLAS_ROUTINE_EACH(DEFINE_WRAP);

}  // namespace phi::dynload
