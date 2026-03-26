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

set -ex

init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'
    SCRIPT_NAME='paddle_musa build script'
    CUR_DIR=$(pwd) 
    PADDLE_PATH=../../Paddle
    PADDLE_PATCHES_DIR=${CUR_DIR}/patches/paddle
    PADDLE_OFFICIAL_UT_DIR=${PADDLE_PATH}/test
}

paddle_musa_ut() {
pushd tests/unittests/collective
python test_broadcast.py
python test_collective_sendrecv_api.py
python test_allreduce.py
python test_collective_scatter.py
python test_allgather.py
python test_c_concat.py
python test_c_identity.py
python test_c_split.py
FLAGS_enable_pir_api=0 FLAGS_enable_pir_in_executor=0 python test_collective_reduce_scatter.py
FLAGS_enable_pir_api=0 FLAGS_enable_pir_in_executor=0 python test_collective_reduce_scatter_api.py
popd

export PYTHONPATH="../../python/tests/"
python tests/unittests/test_flash_attention_musa.py
python tests/unittests/test_take_along_axis_op.py
#python tests/unittests/test_cross_entropy_op.py #need to fix accuracy
pytest tests/unittests/test_matmul_v2_op.py
pytest tests/unittests/test_matmul_op_musa.py
pytest tests/unittests/test_einsum_op_v2_musa.py
pytest tests/unittests/test_softmax_op_musa.py
pytest tests/unittests/test_scatter_op_musa.py
pytest tests/unittests/test_log_softmax_op_dynamic_musa.py
python tests/unittests/test_log_softmax_op_static_musa.py
python tests/unittests/test_weight_only_linear_musa.py
pytest tests/unittests/test_softmax_with_cross_entropy_op_musa.py

pytest tests/unittests/test_group_norm_musa.py::TestGroupNormAPIV2_With_NCHW # only support NCHW layout and fp32
pytest tests/unittests/test_batch_norm_op.py
pytest tests/unittests/test_top_k_v2_op_musa.py
python3 tests/unittests/test_conv2d_op.py
}

main() {
    init

    paddle_musa_ut

    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

    pushd ${PADDLE_OFFICIAL_UT_DIR}/legacy_test
    
    export PYTHONPATH="$PWD"

    pytest test_matmul_out.py
    pytest test_unsqueeze2_op.py
    pytest test_squeeze2_op.py
    pytest test_pool2d_api.py
    pytest test_pool3d_api.py
    pytest test_deformable_conv_v1_op.py
    pytest test_mv_op.py
    pytest test_bmm_op.py
    python3 test_one_hot_v2_op.py #need static from main
    pytest test_argsort_op.py
    pytest test_bilinear_tensor_product_op.py
    pytest test_lamb_op.py
    pytest test_pool1d_api.py
    pytest test_deformable_conv_op.py
    pytest test_baddbmm_op.py
    python test_gru_unit_op.py
    pytest test_lstm_op.py
    pytest test_number_count_op.py
    pytest test_sparse_momentum_op.py
    pytest test_stft_op.py
    # pytest test_cross_entropy2_op.py # need to fix accuracy
    pytest test_shape_op.py 
    pytest test_is_empty_op.py
    pytest test_reverse_op.py
    
    pytest test_reduce_op.py::TestSumOp
    pytest test_reduce_op.py::TestComplexSumOP
    pytest test_reduce_op.py::TestSumOp_ZeroDim
    pytest test_reduce_op.py::TestSumOp5D
    pytest test_reduce_op.py::TestSumOp6D
    pytest test_reduce_op.py::TestSumOp8D
    pytest test_reduce_op.py::TestSumOp_withInt
    pytest test_reduce_op.py::TestSumOp3Dim
    pytest test_reduce_op.py::TestSumOp3D0size
    pytest test_reduce_op.py::TestSumOp3D0size1
    pytest test_reduce_op.py::TestSumOp3D0size2
    pytest test_reduce_op.py::TestSumOp3D0size3
    pytest test_reduce_op.py::TestSumAPIZeroDimKeepDim
    pytest test_mean_op_v1.py
    pytest test_max_min_amax_amin_op.py 

    python test_crop_tensor_op.py
    pytest test_cummax_op.py
    pytest test_cummin_op.py
    pytest test_diagonal_op.py
    pytest test_erf_op.py
    pytest test_fake_dequantize_op.py
    pytest test_fold_op.py
    pytest test_fused_adam_op.py
    pytest test_hinge_loss_op.py
    pytest test_im2sequence_op.py
    pytest test_increment.py
    pytest test_index_put_op.py
    pytest test_index_sample_op.py
    pytest test_label_smooth_op.py
    pytest test_lerp_op.py
    pytest test_masked_fill.py
    pytest test_maxout_op.py
    pytest test_mode_op.py
    pytest test_reduce_as_op.py
    python test_segment_ops.py
    pytest test_shuffle_channel_op.py
    python test_split_op.py
    # pytest test_squared_l2_norm_op.py
    pytest test_stack_op.py
    python test_tile_op.py
    python test_trace_op.py
    pytest test_triu_indices_op.py
    python test_unbind_op.py
    pytest test_unfold_op.py
    pytest test_unique_consecutive_op.py
    pytest test_where_op.py
    pytest test_yolo_box_op.py
    pytest test_empty_op.py
    pytest test_log_loss_op.py
    pytest test_nonzero_api.py
    pytest test_strided_slice_op.py
    pytest test_shuffle_batch_op.py
    pytest test_assign_op.py
    pytest test_add_n_op.py
    python test_cross_op.py
    pytest test_erfinv_op.py
    pytest test_fill_any_op.py
    pytest test_fill_any_like_op.py
    pytest test_fill_constant_op.py
    pytest test_full_op.py
    pytest test_gaussian_random_op.py
    python test_isfinite_v2_op.py
    python test_sign_op.py
    pytest test_lgamma_op.py
    python test_logical_op.py
    # pytest test_pad_op.py
    pytest test_scale_op.py
    pytest test_set_value_op.py
    pytest test_uniform_random_op.py
    pytest test_uniform_random_inplace_op.py
    pytest test_angle_op.py
    pytest test_arange.py
    pytest test_arg_min_max_op.py
    pytest test_beam_search_op.py
    pytest test_broadcast_to_op.py
    pytest test_channel_shuffle.py
    pytest test_clip_op.py
    pytest test_decayed_adagrad_op.py
    pytest test_dequantize_abs_max_op.py
    pytest test_dequantize_log_op.py
    pytest test_diag_embed.py
    pytest test_edit_distance_op.py
    python test_expand_as_v2_op.py
    pytest test_eye_op.py
    pytest test_fill_diagonal_tensor_op.py
    python test_flip.py
    pytest test_frame_op.py
    pytest test_huber_loss_op.py
    pytest test_index_elementwise.py
    pytest test_linspace.py

    python test_bce_loss.py
    pytest test_bitwise_op.py
    pytest test_box_coder_op.py
    pytest test_box_clip_op.py
    pytest test_cvm_op.py
    pytest test_broadcast_tensors_op.py
    pytest test_gammaln_op.py
    pytest test_gumbel_softmax_op.py
    pytest test_cast_op.py
    pytest test_logspace.py
    pytest test_digamma_op.py
    pytest test_multiplex_op.py
    pytest test_nms_op.py
    python test_numel_op.py
    pytest test_overlap_add_op.py
    pytest test_pad3d_op.py
    pytest test_pixel_shuffle_op.py 
    pytest test_pixel_unshuffle.py
    pytest test_polygamma_op.py
    python test_psroi_pool_op.py
    pytest test_range.py
    python test_reshape_op.py
    pytest test_roll_op.py
    pytest test_row_conv_op.py
    pytest test_searchsorted_op.py
    pytest test_share_data_op.py
    pytest test_tril_indices_op.py
    pytest test_trunc_op.py
    pytest test_viterbi_decode_op.py
    pytest test_unstack_op.py
    pytest test_unpool3d_op.py
    pytest test_unpool_op.py
    pytest test_tril_triu_op.py
    pytest test_seed_op.py
    pytest test_renorm_op.py
    pytest test_rrelu_op.py
    pytest test_limit_by_capacity_op.py
    python test_histogram_op.py
    python test_generate_proposals_v2_op.py
    pytest test_unique.py
    python test_atan2_op.py
    pytest test_affine_channel_op.py
    pytest test_accuracy_op.py
    python test_gather_nd_op.py
   
    pytest test_dropout_op.py::TestDropoutOp
    pytest test_dropout_op.py::TestDropoutOpWithSeed
    pytest test_dropout_op.py::TestDropoutOp_ZeroDim
    pytest test_dropout_op.py::TestDropoutCAPI
    pytest test_dropout_op.py::TestDropoutFAPI
    pytest test_dropout_op.py::TestDropout2DCAPI
    pytest test_dropout_op.py::TestDropout2DFAPI
    pytest test_dropout_op.py::TestDropout3DCAPI
    pytest test_dropout_op.py::TestDropout3DFAPI
    pytest test_dropout_op.py::TestAlphaDropoutFAPI
    pytest test_dropout_op.py::TestDropout1DFAPIError
    pytest test_dropout_op.py::TestDropout2DFAPIError
    pytest test_dropout_op.py::TestDropout3DFAPIError
    #pytest test_repeat_interleave_op.py # have passed in musa-4.3.3
    pytest test_gelu_op.py
    pytest test_nn_functional_embedding_dygraph.py
    python test_nn_functional_embedding_static.py
    pytest test_bfloat16_embedding.py
    pytest test_layer_norm_op.py
    pytest test_layer_norm_op_v2.py
    pytest test_top_k_op.py
    pytest test_sync_batch_norm_op_convert.py
    python test_meshgrid_op.py
    pytest test_top_p_sampling.py
    pytest test_fused_bias_act_op.py
    pytest test_rms_norm_op.py -k "not test_rms_norm_backward"
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/collective/fleet
    export PYTHONPATH="$PWD"
    PYTHONPATH=$PYTHONPATH:../../legacy_test pytest test_dgc_momentum_op.py
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/amp
    export PYTHONPATH="$PWD"
    pytest test_amp_api.py
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/ir/pir/translator
    export PYTHONPATH="$PWD"
    pytest test_global_gather_translator.py
    pytest test_nop_translator.py
    pytest test_barrier_translator.py
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/quantization
    export PYTHONPATH="$PWD"
    pytest test_llm_int8_linear.py
    pytest test_weight_only_linear.py
    pytest test_weight_quantize.py
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/contrib
    export PYTHONPATH="$PWD"
    pytest test_correlation.py
    popd

    pushd ${PADDLE_OFFICIAL_UT_DIR}/dygraph_to_static
    export PYTHONPATH="$PWD"
    pytest test_contiguous.py
    popd
}

main $@
