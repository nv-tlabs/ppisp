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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "ppisp_math.cuh"
#include "ppisp_math_bwd.cuh"

// ============================================================================
// Configuration
// ============================================================================

// CUDA kernel launch configuration
constexpr int PPISP_BLOCK_SIZE = 256;

// Helper function to compute grid size
inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// PPISP Forward Kernel
// ============================================================================

__global__ void ppisp_kernel(int batch_size, int num_cameras, int num_frames,
                             const float *__restrict__ exposure_params,
                             const VignettingChannelParams *__restrict__ vignetting_params,
                             const ColorPPISPParams *__restrict__ color_params,
                             const CRFPPISPChannelParams *__restrict__ crf_params,
                             const float3 *__restrict__ rgb_in, float3 *__restrict__ rgb_out,
                             const float2 *__restrict__ pixel_coords, int resolution_x,
                             int resolution_y, int camera_idx, int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    // Load RGB input
    float3 rgb = rgb_in[tid];

    // ISP Pipeline - Full PPISP

    // 1. Exposure compensation
    if (frame_idx != -1) {
        apply_exposure(rgb, exposure_params[frame_idx], rgb);
    }

    // 2. Vignetting correction
    if (camera_idx != -1) {
        float2 pixel_coord;
        if (pixel_coords != nullptr) {
            pixel_coord = pixel_coords[tid];
        } else {
            pixel_coord =
                make_float2(float(tid % resolution_x) + 0.5f, float(tid / resolution_x) + 0.5f);
        }
        apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord, (float)resolution_x,
                         (float)resolution_y, rgb);
    }

    // 3. Color correction (homography)
    if (frame_idx != -1) {
        apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb);
    }

    // 4. Camera Response Function (CRF)
    if (camera_idx != -1) {
        apply_crf_ppisp(rgb, &crf_params[camera_idx * 3], rgb);
    }

    // Store output
    rgb_out[tid] = rgb;
}

// ============================================================================
// PPISP Backward Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void ppisp_bwd_kernel(
    int batch_size, int num_cameras, int num_frames, const float *__restrict__ exposure_params,
    const VignettingChannelParams *__restrict__ vignetting_params,
    const ColorPPISPParams *__restrict__ color_params,
    const CRFPPISPChannelParams *__restrict__ crf_params, const float3 *__restrict__ rgb_in,
    const float3 *__restrict__ rgb_out, const float3 *__restrict__ grad_rgb_out,
    float *__restrict__ grad_exposure_params,
    VignettingChannelParams *__restrict__ grad_vignetting_params,
    ColorPPISPParams *__restrict__ grad_color_params,
    CRFPPISPChannelParams *__restrict__ grad_crf_params, float3 *__restrict__ grad_rgb_in,
    const float2 *__restrict__ pixel_coords, int resolution_x, int resolution_y, int camera_idx,
    int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread gradient accumulators
    float grad_exposure_local = 0.0f;
    VignettingChannelParams grad_vignetting_local[3] = {
        {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    ColorPPISPParams grad_color_local = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    CRFPPISPChannelParams grad_crf_local[3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    if (tid < batch_size) {
        // Load input
        float3 rgb_input = rgb_in[tid];

        // Load or compute pixel coordinate if needed
        float2 pixel_coord = {0.f, 0.f};
        if (camera_idx != -1) {
            if (pixel_coords != nullptr) {
                pixel_coord = pixel_coords[tid];
            } else {
                pixel_coord =
                    make_float2(float(tid % resolution_x) + 0.5f, float(tid / resolution_x) + 0.5f);
            }
        }

        // Recompute forward pass using separate output variables to avoid aliasing
        float3 rgb = rgb_input;
        float3 rgb_after_exp = rgb;
        float3 rgb_after_vig = rgb;
        float3 rgb_after_color = rgb;

        // 1. Exposure
        if (frame_idx != -1) {
            apply_exposure(rgb, exposure_params[frame_idx], rgb_after_exp);
            rgb = rgb_after_exp;
        }

        // 2. Vignetting
        if (camera_idx != -1) {
            apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord,
                             (float)resolution_x, (float)resolution_y, rgb_after_vig);
            rgb = rgb_after_vig;
        } else {
            rgb_after_vig = rgb;
        }

        // 3. Color correction
        if (frame_idx != -1) {
            apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb_after_color);
            rgb = rgb_after_color;
        } else {
            rgb_after_color = rgb;
        }

        // Backward pass (reverse order)
        float3 grad_rgb = grad_rgb_out[tid];

        // 4. CRF backward
        if (camera_idx != -1) {
            apply_crf_ppisp_bwd(rgb_after_color, &crf_params[camera_idx * 3], grad_rgb, grad_rgb,
                                grad_crf_local);
        }

        // 3. Color correction backward
        if (frame_idx != -1) {
            apply_color_correction_ppisp_bwd(rgb_after_vig, &color_params[frame_idx], grad_rgb,
                                             grad_rgb, &grad_color_local);
        }

        // 2. Vignetting backward
        if (camera_idx != -1) {
            apply_vignetting_bwd(rgb_after_exp, &vignetting_params[camera_idx * 3], pixel_coord,
                                 (float)resolution_x, (float)resolution_y, grad_rgb, grad_rgb,
                                 grad_vignetting_local);
        }

        // 1. Exposure backward
        if (frame_idx != -1) {
            apply_exposure_bwd(rgb_input, exposure_params[frame_idx], grad_rgb, grad_rgb,
                               grad_exposure_local);
        }

        // Store RGB input gradient
        grad_rgb_in[tid] = grad_rgb;
    }  // END if (tid < batch_size)

    // Block-level reduction and atomic add for parameter gradients
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    typedef cub::BlockReduce<float2, BLOCK_SIZE> BlockReduceFloat2;

    if (frame_idx != -1) {
        // Exposure
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            float val = BlockReduceFloat(temp).Sum(grad_exposure_local);
            if (threadIdx.x == 0)
                atomicAdd(&grad_exposure_params[frame_idx], val);
        }

        // Color params (4 x float2)
        {
            __shared__ typename BlockReduceFloat2::TempStorage temp;
            ColorPPISPParams *grad_color_out = &grad_color_params[frame_idx];

            float2 val_b = BlockReduceFloat2(temp).Sum(grad_color_local.b);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->b.x, val_b.x);
                atomicAdd(&grad_color_out->b.y, val_b.y);
            }

            float2 val_r = BlockReduceFloat2(temp).Sum(grad_color_local.r);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->r.x, val_r.x);
                atomicAdd(&grad_color_out->r.y, val_r.y);
            }

            float2 val_g = BlockReduceFloat2(temp).Sum(grad_color_local.g);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->g.x, val_g.x);
                atomicAdd(&grad_color_out->g.y, val_g.y);
            }

            float2 val_n = BlockReduceFloat2(temp).Sum(grad_color_local.n);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->n.x, val_n.x);
                atomicAdd(&grad_color_out->n.y, val_n.y);
            }
        }
    }

    if (camera_idx != -1) {
        // Vignetting params (3 channels x 5 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            VignettingChannelParams *grad_vig_out = &grad_vignetting_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_cx = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cx);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cx, val_cx);

                float val_cy = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cy);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cy, val_cy);

                float val_a0 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha0);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha0, val_a0);

                float val_a1 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha1);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha1, val_a1);

                float val_a2 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha2);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha2, val_a2);
            }
        }

        // CRF params (3 channels x 4 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            CRFPPISPChannelParams *grad_crf_out = &grad_crf_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_toe = BlockReduceFloat(temp).Sum(grad_crf_local[ch].toe);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].toe, val_toe);

                float val_shoulder = BlockReduceFloat(temp).Sum(grad_crf_local[ch].shoulder);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].shoulder, val_shoulder);

                float val_gamma = BlockReduceFloat(temp).Sum(grad_crf_local[ch].gamma);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].gamma, val_gamma);

                float val_center = BlockReduceFloat(temp).Sum(grad_crf_local[ch].center);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].center, val_center);
            }
        }
    }
}

// ============================================================================
// Forward Pass Implementation
// ============================================================================

void ppisp_forward(const float *exposure_params, const float *vignetting_params,
                   const float *color_params, const float *crf_params, const float *rgb_in,
                   float *rgb_out, const float *pixel_coords, int num_pixels, int num_cameras,
                   int num_frames, int resolution_w, int resolution_h, int camera_idx,
                   int frame_idx) {
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);

    ppisp_kernel<<<blocks, threads>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<float3 *>(rgb_out),
        reinterpret_cast<const float2 *>(pixel_coords), resolution_w, resolution_h, camera_idx,
        frame_idx);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in ppisp_forward: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Backward Pass Implementation
// ============================================================================

void ppisp_backward(const float *exposure_params, const float *vignetting_params,
                    const float *color_params, const float *crf_params, const float *rgb_in,
                    const float *rgb_out, const float *pixel_coords, const float *v_rgb_out,
                    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
                    float *v_crf_params, float *v_rgb_in, int num_pixels, int num_cameras,
                    int num_frames, int resolution_w, int resolution_h, int camera_idx,
                    int frame_idx) {
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);

    ppisp_bwd_kernel<PPISP_BLOCK_SIZE><<<blocks, threads>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<const float3 *>(rgb_out),
        reinterpret_cast<const float3 *>(v_rgb_out), v_exposure_params,
        reinterpret_cast<VignettingChannelParams *>(v_vignetting_params),
        reinterpret_cast<ColorPPISPParams *>(v_color_params),
        reinterpret_cast<CRFPPISPChannelParams *>(v_crf_params),
        reinterpret_cast<float3 *>(v_rgb_in), reinterpret_cast<const float2 *>(pixel_coords),
        resolution_w, resolution_h, camera_idx, frame_idx);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in ppisp_backward: %s\n", cudaGetErrorString(err));
    }
}
