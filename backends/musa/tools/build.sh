#!/bin/bash
# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'
    SCRIPT_NAME='paddle_musa build script'
    CUR_DIR=$(pwd) 
    PADDLE_PATH=../../Paddle
    PADDLE_PATCHES_DIR=${CUR_DIR}/patches/paddle
}

print_usage() {
  echo -e "\n${RED}Options${NONE}:
  ${BLUE}-a/--all${NONE}: build paddlepaddle and paddle_musa
  ${BLUE}-p/--paddle${NONE}: build paddlepaddle only and install
  ${BLUE}-m/--paddle_musa${NONE}: build paddle_musa only and install
  ${BLUE}-t/--test${NONE}: run all unit test
  ${BLUE}-s/--single_test${NONE}: run single unit test
  ${BLUE}-c/--clean${NONE}: clean paddle_musa
  ${BLUE}-h/--help${NONE}: show usage
  "
}

function copy_impl() {
  file=$1
  paddle_dst_path=$2
  echo -e "${BLUE}copy ${file} to ${PADDLE_PATH} ...${NONE}"
  cp ${CUR_DIR}/${file} ${PADDLE_PATH}/${paddle_dst_path}
  echo -e "${BLUE}copy done ...${NONE}"
}

copy_some_hack_files() {
  copy_impl hack/paddle/phi/kernels/autotune/gpu_timer.h paddle/phi/kernels/autotune/gpu_timer.h
  copy_impl hack/paddle/phi/kernels/group_norm_kernel.h paddle/phi/kernels/group_norm_kernel.h

  copy_impl hack/cuda_hack/float16.h paddle/phi/common/float16.h
  copy_impl hack/cuda_hack/bfloat16.h paddle/phi/common/bfloat16.h
  copy_impl hack/cuda_hack/complex.h paddle/phi/common/complex.h
  
  copy_impl hack/paddle/phi/core/enforce.h paddle/phi/core/enforce.h
  copy_impl hack/paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h
  copy_impl hack/paddle/phi/core/mixed_vector.h paddle/phi/core/mixed_vector.h
  copy_impl hack/paddle/phi/core/mixed_vector.cc paddle/phi/core/mixed_vector.cc
  copy_impl hack/paddle/phi/core/distributed/comm_context_manager.cc paddle/phi/core/distributed/comm_context_manager.cc

  copy_impl hack/paddle/phi/kernels/gpu/reduce_grad.h paddle/phi/kernels/gpu/reduce_grad.h
  copy_impl hack/paddle/phi/kernels/gpu/reduce.h paddle/phi/kernels/gpu/reduce.h
  copy_impl hack/paddle/phi/kernels/gpu/shuffle_batch_utils.h paddle/phi/kernels/gpu/shuffle_batch_utils.h
  copy_impl hack/paddle/phi/kernels/gpu/flash_attn_utils.h paddle/phi/kernels/gpu/flash_attn_utils.h

  copy_impl hack/paddle/phi/kernels/legacy/gpu/layer_norm_cuda_kernel.h paddle/phi/kernels/legacy/gpu/layer_norm_cuda_kernel.h

  copy_impl hack/paddle/phi/kernels/gpudnn/softmax_gpudnn.h paddle/phi/kernels/gpudnn/
  copy_impl hack/paddle/phi/CMakeLists.txt paddle/phi/CMakeLists.txt

  copy_impl hack/python/paddle/distributed/collective.py python/paddle/distributed/collective.py
  copy_impl hack/python/paddle/nn/functional/flash_attention.py python/paddle/nn/functional/flash_attention.py

  copy_impl hack/python/paddle/utils/cpp_extension/__init__.py python/paddle/utils/cpp_extension/__init__.py
  copy_impl hack/python/paddle/utils/cpp_extension/cpp_extension.py python/paddle/utils/cpp_extension/cpp_extension.py
  copy_impl hack/python/paddle/utils/cpp_extension/extension_utils.py python/paddle/utils/cpp_extension/extension_utils.py

  copy_impl hack/paddle/phi/backends/custom/* paddle/phi/backends/custom/
  copy_impl hack/paddle/phi/backends/gpu/* paddle/phi/backends/gpu/
  copy_impl hack/paddle/phi/common/* paddle/phi/common/
  copy_impl hack/paddle/phi/kernels/funcs/*.h paddle/phi/kernels/funcs/
  copy_impl hack/paddle/phi/kernels/funcs/blas/*.h paddle/phi/kernels/funcs/blas/
  copy_impl hack/paddle/phi/kernels/funcs/detail/*.h paddle/phi/kernels/funcs/detail/
  copy_impl hack/paddle/phi/kernels/sparse/gpu/* paddle/phi/kernels/sparse/gpu/
  copy_impl hack/paddle/phi/kernels/funcs/sparse/* paddle/phi/kernels/funcs/sparse/
  copy_impl hack/paddle/phi/kernels/gpudnn/softmax_gpudnn.h paddle/phi/kernels/gpudnn/
  copy_impl hack/paddle/phi/kernels/impl/*.h paddle/phi/kernels/impl/
  copy_impl hack/paddle/phi/backends/dynload/*.h paddle/phi/backends/dynload/
  copy_impl hack/paddle/phi/kernels/*.cc paddle/phi/kernels/
  copy_impl hack/paddle/phi/kernels/*.h paddle/phi/kernels/

  copy_impl hack/test/legacy_test/auto_parallel_op_test.py test/legacy_test/auto_parallel_op_test.py
  copy_impl hack/test/legacy_test/op_test.py test/legacy_test/op_test.py
}

post_copy_some_hack_files() {
  copy_impl hack/third_party/warprnnt/include/*.h third_party/warprnnt/include/
  copy_impl hack/third_party/warpctc/include/*.h third_party/warpctc/include/
}

apply_paddle_patches() {
  # apply patches into paddlepaddle 
  echo -e "${BLUE}Applying patches to ${PADDLE_PATH} ...${NONE}"
  # clean PyTorch before patching
  if [ -d "$PADDLE_PATH/.git" ]; then
    echo -e "${BLUE}Stash and checkout the paddle environment before patching. ${NONE}"
    pushd $PADDLE_PATH
    git stash -u
    popd
  fi

  for file in $(find ${PADDLE_PATCHES_DIR} -type f -print); do
    if [ "${file##*.}"x = "patch"x ]; then
      echo -e "${BLUE}applying patch: $file ${NONE}"
      pushd $PADDLE_PATH
      git apply --check $file
      git apply $file
      popd
    fi
  done
  
  copy_some_hack_files
}

build_paddlepaddle() {
  pushd $PADDLE_PATH
  if [ ! -f "CMakeLists.txt" ];then
    git submodule update --init --recursive --jobs 1;get_pd_ret=$?
    if [ "$get_pd_ret" != 0 ];then
        echo "get paddlepaddle failed!!!!"
        exit 11
    fi
  fi 
  
  mkdir -p build
  pushd build
  
  apply_paddle_patches;apply_ret=$?
  if [ "$apply_ret" != 0 ];then
      echo "apply paddle patches error"
      exit 10
  fi
  if [ ! -d "CMakeFiles" ];then 
    cmake .. -DWITH_MKL=ON \
            -DWITH_GPU=OFF \
            -DPY_VERSION=3.10 \
            -DWITH_CINN=OFF \
            -DWITH_DISTRIBUTE=ON \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
            -DCMAKE_CXX_FLAGS="-I/usr/local/musa/include";cmake_ret=$? 
    if [ "$cmake_ret" != 0 ];then
        echo "cmake error"
        exit 9
    fi
  fi
  
  make -j 128;make_ret=$?
  if [ "$cmake_ret" != 0 ];then
      echo "paddle make have error, ret=${make_ret}"
  fi

  if [ ! -f "python/dist/$(ls python/dist)" ]; then
      echo "paddle build failed!!!!!!!"
      exit 12
  fi
  pip uninstall paddlepaddle
  pip install python/dist/paddlepaddle*.whl --force-reinstall
  
  post_copy_some_hack_files #TODO(moore threads): replace by implemention in cmake
  
  popd
  popd  
}

build_paddle_musa() {
  
  bash tools/compile.sh;build_ret=$?
  if [ "$build_ret" != 0 ];then
      echo "CMake Error Found !!!"
      exit 8;
  fi
  
  pip uninstall paddle_musa
  pip install build/dist/paddle_musa-*.whl

  PADDLE_MUSA_ROOT_PATH="$(cd ../../ && pwd)" python setup_ops.py install
}

run_all_ut() {
  export PYTHONPATH="$(pwd)/tests/unittests:${PYTHONPATH}"
  for file in $(ls tests/unittests/test_*.py | grep -v "test_reduce_op.py" | grep -v "test_reshape_op.py"); do
    echo "Running $file"
    python3 -m unittest "$file"
  done

  bash tools/run_ut.sh
}

run_single_ut() {
  echo "-s unimplement"
}

clean() {

  # clean paddlepaddle
  echo $BULE"begin clean paddlepaddle. "$NONE
  pushd $PADDLE_PATH
  rm build -rf
  popd
  echo $BULE"clean paddlepaddle finished. "$NONE

  # clean paddle_musa
  echo $BULE"begin clean paddle_musa. "$NONE
  rm build -rf 
  echo $BULE"clean paddle_musa finished. "$NONE

}

main() {
  init
  while true; do
    case "$1" in
    -a | --all)
      build_paddlepaddle
      build_paddle_musa 
      shift
      ;;
    -p | --paddle)
      build_paddlepaddle
      shift
      ;;
    -m | --paddle_musa)
      build_paddle_musa
      shift
      ;;
    -t | --test)
      run_all_ut
      shift
      ;;
    -s | --single_test)
      run_single_ut
      shift
      ;;
    -c | --clean)
      clean
      shift
      ;;
    -h | --help)
      print_usage
      exit
      ;;
    --)
      shift
      break
      ;;
    *)
      print_usage
      exit 0
      ;;
    esac
  done
}


main $@
