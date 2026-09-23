# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient checking tests using torch.autograd.gradcheck.

Validates CUDA gradient implementations against numerical finite differences.
These tests complement the CUDA-vs-PyTorch comparison tests by verifying
gradients through an independent numerical method.

Note: The CUDA kernel uses float32 internally, so gradcheck tolerances are
set accordingly. Tests use float64 for numerical differentiation precision
but actual kernel computations are in float32.
"""

import torch
import pytest
from torch.autograd import gradcheck

import ppisp_cuda


# Tolerances for float32 precision
GRADCHECK_EPS = 1e-3     # Perturbation for finite differences
GRADCHECK_ATOL = 5e-3    # Absolute tolerance
GRADCHECK_RTOL = 5e-2    # Relative tolerance (5%)


class PPISPAutograd(torch.autograd.Function):
    """Autograd wrapper for CUDA PPISP forward/backward."""

    @staticmethod
    def forward(
        ctx,
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    ):
        # Ensure contiguous float32
        exposure_params = exposure_params.float().contiguous()
        vignetting_params = vignetting_params.float().contiguous()
        color_params = color_params.float().contiguous()
        crf_params = crf_params.float().contiguous()
        rgb_in = rgb_in.float().contiguous()
        pixel_coords = pixel_coords.float().contiguous()

        rgb_out = ppisp_cuda.ppisp_forward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            pixel_coords,
            resolution_w,
            resolution_h,
            camera_idx,
            frame_idx,
        )

        ctx.save_for_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, pixel_coords
        )
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h
        ctx.camera_idx = camera_idx
        ctx.frame_idx = frame_idx

        return rgb_out

    @staticmethod
    def backward(ctx, v_rgb_out):
        (exposure_params, vignetting_params,
         color_params, crf_params, rgb_in, pixel_coords) = ctx.saved_tensors

        (v_exposure_params, v_vignetting_params,
         v_color_params, v_crf_params, v_rgb_in) = ppisp_cuda.ppisp_backward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            pixel_coords,
            v_rgb_out.contiguous(),
            ctx.resolution_w,
            ctx.resolution_h,
            ctx.camera_idx,
            ctx.frame_idx,
        )

        return (
            v_exposure_params,
            v_vignetting_params,
            v_color_params,
            v_crf_params,
            v_rgb_in,
            None,  # pixel_coords
            None,  # resolution_w
            None,  # resolution_h
            None,  # camera_idx
            None,  # frame_idx
        )


def create_test_inputs(
    num_pixels=5,
    num_cameras=1,
    num_frames=1,
    resolution_w=64,
    resolution_h=64,
    camera_idx=0,
    frame_idx=0,
    requires_grad=True,
    device='cuda',
    dtype=torch.float64,
    seed=42,
):
    """Create test inputs for gradient checking.

    Uses small parameter perturbations around identity to avoid numerical issues.
    Note: CUDA kernel internally uses float32, so dtype=float64 only affects
    gradcheck's numerical differentiation precision.
    """
    torch.manual_seed(seed)

    exposure_params = torch.randn(
        num_frames, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.1

    vignetting_params = torch.randn(
        num_cameras, 3, 5, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.01

    # Color params: very small perturbations to keep homography near identity
    color_params = torch.randn(
        num_frames, 8, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.02

    # CRF params: initialized near identity
    crf_params = torch.randn(
        num_cameras, 3, 4, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.1

    # RGB input in valid range [0.2, 0.8] to avoid edge cases in CRF
    rgb_in = torch.rand(
        num_pixels, 3, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.5 + 0.25

    # Pixel coordinates - avoid extreme edges for vignetting
    pixel_coords = torch.rand(
        num_pixels, 2, device=device, dtype=dtype, requires_grad=False
    )
    pixel_coords[:, 0] = pixel_coords[:, 0] * 0.6 + 0.2  # 20%-80% of width
    pixel_coords[:, 1] = pixel_coords[:, 1] * 0.6 + 0.2  # 20%-80% of height
    pixel_coords[:, 0] *= resolution_w
    pixel_coords[:, 1] *= resolution_h

    return {
        'exposure_params': exposure_params,
        'vignetting_params': vignetting_params,
        'color_params': color_params,
        'crf_params': crf_params,
        'rgb_in': rgb_in,
        'pixel_coords': pixel_coords,
        'resolution_w': resolution_w,
        'resolution_h': resolution_h,
        'camera_idx': camera_idx,
        'frame_idx': frame_idx,
    }


def ppisp_wrapper(
    exposure_params,
    vignetting_params,
    color_params,
    crf_params,
    rgb_in,
    pixel_coords,
    resolution_w,
    resolution_h,
    camera_idx,
    frame_idx,
):
    """Wrapper for gradcheck that handles double->float conversion."""
    return PPISPAutograd.apply(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    )


def test_gradcheck_all_params():
    """Test gradients for all parameters simultaneously using finite differences.

    This is the primary numerical gradient verification test. It validates that
    the CUDA backward pass produces gradients consistent with numerical
    differentiation for all parameter groups at once.
    """
    inputs = create_test_inputs(num_pixels=3)

    def func(exp, vig, col, crf, rgb):
        return ppisp_wrapper(
            exp, vig, col, crf, rgb,
            inputs['pixel_coords'],
            inputs['resolution_w'],
            inputs['resolution_h'],
            inputs['camera_idx'],
            inputs['frame_idx'],
        )

    assert gradcheck(
        func,
        (
            inputs['exposure_params'],
            inputs['vignetting_params'],
            inputs['color_params'],
            inputs['crf_params'],
            inputs['rgb_in'],
        ),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        raise_exception=True,
    )


def test_gradient_magnitude_reasonable():
    """Verify gradient magnitudes are reasonable (not exploding/vanishing).

    This sanity check ensures gradients are finite and within expected bounds,
    catching issues like NaN propagation or gradient explosion.
    """
    torch.manual_seed(42)

    num_pixels = 20
    resolution_w, resolution_h = 64, 64

    # Create leaf tensors
    exposure_params = torch.empty(
        1, device='cuda', dtype=torch.float32
    ).normal_(0, 0.1).requires_grad_(True)

    vignetting_params = torch.empty(
        1, 3, 5, device='cuda', dtype=torch.float32
    ).normal_(0, 0.01).requires_grad_(True)

    color_params = torch.empty(
        1, 8, device='cuda', dtype=torch.float32
    ).normal_(0, 0.02).requires_grad_(True)

    crf_params = torch.empty(
        1, 3, 4, device='cuda', dtype=torch.float32
    ).normal_(0, 0.1).requires_grad_(True)

    rgb_in = torch.empty(
        num_pixels, 3, device='cuda', dtype=torch.float32
    ).uniform_(0.25, 0.75).requires_grad_(True)

    pixel_coords = torch.empty(
        num_pixels, 2, device='cuda', dtype=torch.float32
    ).uniform_(0, 1)
    pixel_coords[:, 0] = pixel_coords[:, 0] * resolution_w
    pixel_coords[:, 1] = pixel_coords[:, 1] * resolution_h

    rgb_out = ppisp_wrapper(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        0,  # camera_idx
        0,  # frame_idx
    )

    # Backward with unit gradient
    rgb_out.sum().backward()

    # Check gradients exist and are finite
    for name, param in [
        ('exposure_params', exposure_params),
        ('vignetting_params', vignetting_params),
        ('color_params', color_params),
        ('crf_params', crf_params),
        ('rgb_in', rgb_in),
    ]:
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(
        ), f"{name} has non-finite gradient"

        grad_norm = param.grad.norm().item()
        assert grad_norm < 1e6, f"{name} gradient too large: {grad_norm}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
