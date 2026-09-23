/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _PPISP_BINDINGS_H_INC
#define _PPISP_BINDINGS_H_INC

#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/SmallVector.h>
#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>

#include "src/ppisp_constants.h"

// =============================================================================
// Forward pass for PPISP image processing
// =============================================================================

void ppisp_forward(
    // Parameters (per-camera/per-frame)
    const float *exposure_params,    // [num_frames]
    const float *vignetting_params,  // [num_cameras, 3, 5]
    const float *color_params,       // [num_frames, 8]
    const float *crf_params,         // [num_cameras, 3, 4]
    // Input/Output
    const float *rgb_in,        // [num_pixels, 3]
    float *rgb_out,             // [num_pixels, 3]
    const float *pixel_coords,  // [num_pixels, 2] or nullptr
    // Dimensions
    int num_pixels, int num_cameras, int num_frames, int resolution_w, int resolution_h,
    int camera_idx, int frame_idx);

// =============================================================================
// Backward pass for PPISP image processing
// =============================================================================

void ppisp_backward(
    // Parameters (per-camera/per-frame)
    const float *exposure_params, const float *vignetting_params, const float *color_params,
    const float *crf_params,
    // Input from forward
    const float *rgb_in, const float *pixel_coords,
    // Gradient of loss w.r.t. output
    const float *v_rgb_out,
    // Gradients w.r.t. parameters
    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
    float *v_crf_params, float *v_rgb_in,
    // Dimensions
    int num_pixels, int num_cameras, int num_frames, int resolution_w, int resolution_h,
    int camera_idx, int frame_idx);

// =============================================================================
// Forward/backward for PPISP regularization loss
// =============================================================================

void ppisp_regularization_forward(
    // Parameters (per-frame/per-camera)
    const float *exposure_params,    // [num_frames]
    const float *vignetting_params,  // [num_cameras, 3, 5]
    const float *color_params,       // [num_frames, 8]
    const float *crf_params,         // [num_cameras, 3, 4]
    // Outputs
    float *loss_out,         // scalar
    float *frame_mean_sums,  // [PPISP_FRAME_MEAN_SUMS_SIZE]
    // Dimensions
    int num_cameras, int num_frames,
    // Weights
    float exposure_mean_weight, float vig_center_weight,
    float vig_channel_weight, float vig_non_pos_weight, float color_mean_weight,
    float crf_channel_weight);

void ppisp_regularization_backward(
    // Camera parameters; the frame terms only need the saved frame_mean_sums
    const float *vignetting_params,  // [num_cameras, 3, 5]
    const float *crf_params,         // [num_cameras, 3, 4]
    // Upstream gradient
    const float *grad_loss,  // scalar
    // Gradients w.r.t. parameters
    float *grad_exposure_params,    // [num_frames]
    float *grad_vignetting_params,  // [num_cameras, 3, 5]
    float *grad_color_params,       // [num_frames, 8]
    float *grad_crf_params,         // [num_cameras, 3, 4]
    // Saved forward output
    float *frame_mean_sums,  // [PPISP_FRAME_MEAN_SUMS_SIZE]
    // Dimensions
    int num_cameras, int num_frames,
    // Weights
    float exposure_mean_weight,
    float vig_center_weight, float vig_channel_weight, float vig_non_pos_weight,
    float color_mean_weight, float crf_channel_weight);

// =============================================================================
// PyTorch tensor wrappers
// =============================================================================

// Input validation is opt-in: the checks below would run on every wrapper call,
// so they are off by default and ppisp.set_validate_inputs(True) enables them
// for debugging. Unchecked malformed inputs reach the kernels as-is.
inline std::atomic<bool> &ppisp_validate_inputs() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

// The kernels read raw float pointers with hard-coded strides, so every tensor
// must be float32, contiguous, and of the shape the kernel assumes. The ATen
// checks raise RuntimeError with the argument name and position.
inline void ppisp_check_tensor(at::CheckedFrom c, const at::TensorArg &t,
                               at::IntArrayRef sizes) {
    at::checkScalarType(c, t, at::kFloat);
    at::checkContiguous(c, t);
    at::checkSize(c, t, sizes);
}

// The four parameter tensors: exposure [num_frames], vignetting
// [num_cameras, 3, 5], color [num_frames, 8], crf [num_cameras, 3, 4]. The two
// frame counts and the two camera counts must agree. Appends their TensorArgs
// (#1-#4) to `all` for the same-GPU check and returns the counts.
inline std::pair<int64_t, int64_t> ppisp_check_params(
    at::CheckedFrom c, const torch::Tensor &exposure_params,
    const torch::Tensor &vignetting_params, const torch::Tensor &color_params,
    const torch::Tensor &crf_params, c10::SmallVectorImpl<at::TensorArg> &all) {
    const at::TensorArg exposure{exposure_params, "exposure_params", 1};
    const at::TensorArg vignetting{vignetting_params, "vignetting_params", 2};
    const at::TensorArg color{color_params, "color_params", 3};
    const at::TensorArg crf{crf_params, "crf_params", 4};
    at::checkDim(c, exposure, 1);
    at::checkDim(c, crf, 3);
    const int64_t num_frames = exposure->size(0);
    const int64_t num_cameras = crf->size(0);
    ppisp_check_tensor(c, exposure, {num_frames});
    ppisp_check_tensor(c, color, {num_frames, PPISP_COLOR_PARAMS});
    ppisp_check_tensor(c, crf, {num_cameras, 3, PPISP_CRF_PARAMS_PER_CHANNEL});
    ppisp_check_tensor(c, vignetting, {num_cameras, 3, PPISP_VIGNETTING_PARAMS_PER_CHANNEL});
    all.append({exposure, vignetting, color, crf});
    return {num_cameras, num_frames};
}

// -1 disables a parameter group; any other index must address a row.
inline void ppisp_check_indices(int camera_idx, int64_t num_cameras, int frame_idx,
                                int64_t num_frames) {
    TORCH_CHECK_INDEX(camera_idx == -1 || (camera_idx >= 0 && camera_idx < num_cameras),
                      "ppisp: camera_idx ", camera_idx, " out of range for ", num_cameras,
                      " cameras");
    TORCH_CHECK_INDEX(frame_idx == -1 || (frame_idx >= 0 && frame_idx < num_frames),
                      "ppisp: frame_idx ", frame_idx, " out of range for ", num_frames,
                      " frames");
}

// The kernels load color_params as ColorPPISPParams, whose float2 members need
// 8-byte alignment, and pixel_coords as float2. A contiguous view at an odd
// float offset, such as a slice of a flat buffer, is copied into an aligned
// allocation.
inline torch::Tensor ppisp_float2_aligned(torch::Tensor tensor) {
    constexpr uintptr_t kFloat2Align = 2 * sizeof(float);
    if (reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kFloat2Align != 0) {
        return tensor.clone();
    }
    return tensor;
}

// Validation shared by the image forward and backward: parameters, the pixel
// tensors and their count, one CUDA device for everything, and the indices.
inline void ppisp_check_image_inputs(
    at::CheckedFrom c, const torch::Tensor &exposure_params,
    const torch::Tensor &vignetting_params, const torch::Tensor &color_params,
    const torch::Tensor &crf_params, const torch::Tensor &rgb_in,
    const c10::optional<torch::Tensor> &pixel_coords, const torch::Tensor *v_rgb_out,
    int camera_idx, int frame_idx) {
    c10::SmallVector<at::TensorArg, 7> all;
    const auto counts = ppisp_check_params(c, exposure_params, vignetting_params, color_params,
                                           crf_params, all);
    const at::TensorArg rgb_arg{rgb_in, "rgb_in", 5};
    at::checkDim(c, rgb_arg, 2);
    const int64_t num_pixels = rgb_in.size(0);
    // The kernels index pixels with int: the forward computes blocks with divUp in
    // int, and the backward's grid-stride loop counter runs past the end by up to
    // one stride, which never exceeds the pixel count rounded up to a block, so
    // keep the count in half range.
    TORCH_CHECK(num_pixels <= std::numeric_limits<int>::max() / 2,
                "ppisp: too many pixels for int indexing: ", num_pixels);
    ppisp_check_tensor(c, rgb_arg, {num_pixels, 3});
    all.push_back(rgb_arg);
    if (pixel_coords.has_value()) {
        all.emplace_back(*pixel_coords, "pixel_coords", 6);
        ppisp_check_tensor(c, all.back(), {num_pixels, 2});
    }
    if (v_rgb_out != nullptr) {
        all.emplace_back(*v_rgb_out, "v_rgb_out", 7);
        ppisp_check_tensor(c, all.back(), {num_pixels, 3});
    }
    at::checkAllSameGPU(c, all);
    ppisp_check_indices(camera_idx, counts.first, frame_idx, counts.second);
}

// Validation shared by the regularization forward and backward: parameters, the
// backward's grad_loss and frame_mean_sums when given, and one CUDA device.
inline void ppisp_check_regularization_inputs(
    at::CheckedFrom c, const torch::Tensor &exposure_params,
    const torch::Tensor &vignetting_params, const torch::Tensor &color_params,
    const torch::Tensor &crf_params, const torch::Tensor *grad_loss,
    const torch::Tensor *frame_mean_sums) {
    c10::SmallVector<at::TensorArg, 6> all;
    ppisp_check_params(c, exposure_params, vignetting_params, color_params, crf_params, all);
    if (grad_loss != nullptr) {
        all.emplace_back(*grad_loss, "grad_loss", 5);
        at::checkScalarType(c, all.back(), at::kFloat);
        at::checkNumel(c, all.back(), 1);
    }
    if (frame_mean_sums != nullptr) {
        all.emplace_back(*frame_mean_sums, "frame_mean_sums", 6);
        ppisp_check_tensor(c, all.back(), {PPISP_FRAME_MEAN_SUMS_SIZE});
    }
    at::checkAllSameGPU(c, all);
}

torch::Tensor ppisp_forward_tensor(torch::Tensor exposure_params,    // [num_frames]
                                   torch::Tensor vignetting_params,  // [num_cameras, 3, 5]
                                   torch::Tensor color_params,       // [num_frames, 8]
                                   torch::Tensor crf_params,         // [num_cameras, 3, 4]
                                   torch::Tensor rgb_in,             // [num_pixels, 3]
                                   c10::optional<torch::Tensor> pixel_coords,  // [num_pixels, 2]
                                   int resolution_w, int resolution_h, int camera_idx,
                                   int frame_idx) {
    if (ppisp_validate_inputs().load(std::memory_order_relaxed)) {
        ppisp_check_image_inputs("ppisp_forward", exposure_params, vignetting_params, color_params,
                                 crf_params, rgb_in, pixel_coords, nullptr, camera_idx,
                                 frame_idx);
    }
    // Select the device of the inputs so the current-stream lookup and the
    // launch target it, without a Python-side context manager.
    const c10::cuda::CUDAGuard device_guard(rgb_in.device());
    color_params = ppisp_float2_aligned(color_params);
    if (pixel_coords.has_value()) {
        pixel_coords = ppisp_float2_aligned(*pixel_coords);
    }
    int num_pixels = rgb_in.size(0);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto rgb_out = torch::empty_like(rgb_in);

    ppisp_forward(exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
                  color_params.data_ptr<float>(), crf_params.data_ptr<float>(),
                  rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(),
                  pixel_coords.has_value() ? pixel_coords->data_ptr<float>() : nullptr, num_pixels,
                  num_cameras, num_frames, resolution_w, resolution_h, camera_idx, frame_idx);

    return rgb_out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ppisp_backward_tensor(torch::Tensor exposure_params, torch::Tensor vignetting_params,
                      torch::Tensor color_params, torch::Tensor crf_params, torch::Tensor rgb_in,
                      c10::optional<torch::Tensor> pixel_coords,
                      torch::Tensor v_rgb_out, int resolution_w, int resolution_h, int camera_idx,
                      int frame_idx) {
    if (ppisp_validate_inputs().load(std::memory_order_relaxed)) {
        ppisp_check_image_inputs("ppisp_backward", exposure_params, vignetting_params, color_params,
                                 crf_params, rgb_in, pixel_coords, &v_rgb_out, camera_idx,
                                 frame_idx);
    }
    const c10::cuda::CUDAGuard device_guard(rgb_in.device());
    color_params = ppisp_float2_aligned(color_params);
    if (pixel_coords.has_value()) {
        pixel_coords = ppisp_float2_aligned(*pixel_coords);
    }
    int num_pixels = rgb_in.size(0);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto v_exposure_params = torch::zeros_like(exposure_params);
    auto v_vignetting_params = torch::zeros_like(vignetting_params);
    auto v_color_params = torch::zeros_like(color_params);
    auto v_crf_params = torch::zeros_like(crf_params);
    auto v_rgb_in = torch::empty_like(rgb_in);  // fully written by the kernel

    ppisp_backward(exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
                   color_params.data_ptr<float>(), crf_params.data_ptr<float>(),
                   rgb_in.data_ptr<float>(),
                   pixel_coords.has_value() ? pixel_coords->data_ptr<float>() : nullptr,
                   v_rgb_out.data_ptr<float>(), v_exposure_params.data_ptr<float>(),
                   v_vignetting_params.data_ptr<float>(), v_color_params.data_ptr<float>(),
                   v_crf_params.data_ptr<float>(), v_rgb_in.data_ptr<float>(), num_pixels,
                   num_cameras, num_frames, resolution_w, resolution_h, camera_idx, frame_idx);

    return std::make_tuple(v_exposure_params, v_vignetting_params, v_color_params, v_crf_params,
                           v_rgb_in);
}

std::tuple<torch::Tensor, torch::Tensor> ppisp_regularization_forward_tensor(
    torch::Tensor exposure_params,    // [num_frames]
    torch::Tensor vignetting_params,  // [num_cameras, 3, 5]
    torch::Tensor color_params,       // [num_frames, 8]
    torch::Tensor crf_params,         // [num_cameras, 3, 4]
    float exposure_mean_weight, float vig_center_weight,
    float vig_channel_weight, float vig_non_pos_weight, float color_mean_weight,
    float crf_channel_weight) {
    if (ppisp_validate_inputs().load(std::memory_order_relaxed)) {
        ppisp_check_regularization_inputs("ppisp_regularization_forward", exposure_params,
                                          vignetting_params, color_params, crf_params, nullptr,
                                          nullptr);
    }
    const c10::cuda::CUDAGuard device_guard(exposure_params.device());
    color_params = ppisp_float2_aligned(color_params);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    // Both outputs are fully written by the kernel, including for empty inputs.
    auto loss = torch::empty({}, exposure_params.options());
    auto frame_mean_sums = torch::empty({PPISP_FRAME_MEAN_SUMS_SIZE}, exposure_params.options());

    ppisp_regularization_forward(
        exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
        color_params.data_ptr<float>(), crf_params.data_ptr<float>(), loss.data_ptr<float>(),
        frame_mean_sums.data_ptr<float>(), num_cameras, num_frames, exposure_mean_weight,
        vig_center_weight, vig_channel_weight, vig_non_pos_weight, color_mean_weight,
        crf_channel_weight);

    return std::make_tuple(loss, frame_mean_sums);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ppisp_regularization_backward_tensor(
    torch::Tensor exposure_params,    // [num_frames]
    torch::Tensor vignetting_params,  // [num_cameras, 3, 5]
    torch::Tensor color_params,       // [num_frames, 8]
    torch::Tensor crf_params,         // [num_cameras, 3, 4]
    torch::Tensor grad_loss,          // scalar
    torch::Tensor frame_mean_sums,    // [PPISP_FRAME_MEAN_SUMS_SIZE]
    float exposure_mean_weight, float vig_center_weight, float vig_channel_weight,
    float vig_non_pos_weight, float color_mean_weight, float crf_channel_weight) {
    if (ppisp_validate_inputs().load(std::memory_order_relaxed)) {
        ppisp_check_regularization_inputs("ppisp_regularization_backward", exposure_params,
                                          vignetting_params, color_params, crf_params,
                                          &grad_loss, &frame_mean_sums);
    }
    const c10::cuda::CUDAGuard device_guard(exposure_params.device());
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto grad_loss_contig = grad_loss.contiguous();
    // Every element is assigned by the backward kernel, zero for disabled terms.
    auto grad_exposure_params = torch::empty_like(exposure_params);
    auto grad_vignetting_params = torch::empty_like(vignetting_params);
    auto grad_color_params = torch::empty_like(color_params);
    auto grad_crf_params = torch::empty_like(crf_params);

    ppisp_regularization_backward(
        vignetting_params.data_ptr<float>(), crf_params.data_ptr<float>(),
        grad_loss_contig.data_ptr<float>(),
        grad_exposure_params.data_ptr<float>(), grad_vignetting_params.data_ptr<float>(),
        grad_color_params.data_ptr<float>(), grad_crf_params.data_ptr<float>(),
        frame_mean_sums.data_ptr<float>(), num_cameras, num_frames, exposure_mean_weight,
        vig_center_weight, vig_channel_weight, vig_non_pos_weight, color_mean_weight,
        crf_channel_weight);

    return std::make_tuple(grad_exposure_params, grad_vignetting_params, grad_color_params,
                           grad_crf_params);
}

#endif  // _PPISP_BINDINGS_H_INC
