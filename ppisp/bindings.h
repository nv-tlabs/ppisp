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

#include <torch/extension.h>

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
    // Input/Output from forward
    const float *rgb_in, const float *rgb_out, const float *pixel_coords,
    // Gradient of loss w.r.t. output
    const float *v_rgb_out,
    // Gradients w.r.t. parameters
    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
    float *v_crf_params, float *v_rgb_in,
    // Dimensions
    int num_pixels, int num_cameras, int num_frames, int resolution_w, int resolution_h,
    int camera_idx, int frame_idx);

// =============================================================================
// PyTorch tensor wrappers
// =============================================================================

torch::Tensor ppisp_forward_tensor(torch::Tensor exposure_params,    // [num_frames]
                                   torch::Tensor vignetting_params,  // [num_cameras, 3, 5]
                                   torch::Tensor color_params,       // [num_frames, 8]
                                   torch::Tensor crf_params,         // [num_cameras, 3, 4]
                                   torch::Tensor rgb_in,             // [num_pixels, 3]
                                   c10::optional<torch::Tensor> pixel_coords,  // [num_pixels, 2]
                                   int resolution_w, int resolution_h, int camera_idx,
                                   int frame_idx) {
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
                      torch::Tensor rgb_out, c10::optional<torch::Tensor> pixel_coords,
                      torch::Tensor v_rgb_out, int resolution_w, int resolution_h, int camera_idx,
                      int frame_idx) {
    int num_pixels = rgb_in.size(0);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto v_exposure_params = torch::zeros_like(exposure_params);
    auto v_vignetting_params = torch::zeros_like(vignetting_params);
    auto v_color_params = torch::zeros_like(color_params);
    auto v_crf_params = torch::zeros_like(crf_params);
    auto v_rgb_in = torch::zeros_like(rgb_in);

    ppisp_backward(exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
                   color_params.data_ptr<float>(), crf_params.data_ptr<float>(),
                   rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(),
                   pixel_coords.has_value() ? pixel_coords->data_ptr<float>() : nullptr,
                   v_rgb_out.data_ptr<float>(), v_exposure_params.data_ptr<float>(),
                   v_vignetting_params.data_ptr<float>(), v_color_params.data_ptr<float>(),
                   v_crf_params.data_ptr<float>(), v_rgb_in.data_ptr<float>(), num_pixels,
                   num_cameras, num_frames, resolution_w, resolution_h, camera_idx, frame_idx);

    return std::make_tuple(v_exposure_params, v_vignetting_params, v_color_params, v_crf_params,
                           v_rgb_in);
}

#endif  // _PPISP_BINDINGS_H_INC
