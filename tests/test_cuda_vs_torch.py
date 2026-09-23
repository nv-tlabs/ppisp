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

"""
Test to validate CUDA backend numerical equivalence with Torch backend.

This test ensures that the CUDA implementation produces identical results
to the pure PyTorch reference implementation for both forward and backward passes.
"""

import torch
import pytest
import numpy as np
from typing import Dict, Tuple

import ppisp_cuda
import ppisp
from tests.torch_reference import ppisp_apply_torch


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_params(
    num_cameras: int = 2,
    num_frames: int = 5,
    seed: int = 42,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create PPISP parameters with random values.

    Parameters are kept small to ensure numerical stability.
    """
    torch.manual_seed(seed)

    # Exposure: [num_frames] - small range to keep output in valid range
    exposure_params = torch.empty(
        num_frames, device=device, dtype=dtype
    ).uniform_(-0.5, 0.5).requires_grad_(True)

    # Vignetting: [num_cameras, 3, 5] - small polynomial coefficients
    vignetting_params = torch.empty(
        num_cameras, 3, 5, device=device, dtype=dtype
    ).uniform_(-0.1, 0.1).requires_grad_(True)

    # Color correction: [num_frames, 8] - small latent offsets
    color_params = torch.empty(
        num_frames, 8, device=device, dtype=dtype
    ).uniform_(-0.1, 0.1).requires_grad_(True)

    # CRF parameters: [num_cameras, 3, 4] - small raw parameter values
    crf_params = torch.empty(
        num_cameras, 3, 4, device=device, dtype=dtype
    ).uniform_(-0.5, 0.5).requires_grad_(True)

    return {
        'exposure_params': exposure_params,
        'vignetting_params': vignetting_params,
        'color_params': color_params,
        'crf_params': crf_params,
    }


def create_test_inputs(
    batch_size: int = 256,
    resolution_w: int = 512,
    resolution_h: int = 512,
    num_cameras: int = 2,
    num_frames: int = 5,
    seed: int = 42,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create consistent test inputs.

    RGB values are kept in [0.1, 0.9] range to avoid edge cases in CRF.
    Pixel coordinates avoid edges for stable vignetting gradients.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random RGB values in [0.1, 0.9] range (avoid edge cases)
    rgb = torch.empty(batch_size, 3, device=device,
                      dtype=dtype).uniform_(0.1, 0.9)

    # Pixel coordinates - avoid edges for stable vignetting
    pixel_coords = torch.empty(batch_size, 2, device=device, dtype=dtype)
    pixel_coords[:, 0].uniform_(0.1 * resolution_w, 0.9 * resolution_w)
    pixel_coords[:, 1].uniform_(0.1 * resolution_h, 0.9 * resolution_h)

    # Random camera and frame indices
    camera_idx = torch.randint(0, num_cameras, (1,)).item()
    frame_idx = torch.randint(0, num_frames, (1,)).item()

    return {
        'rgb': rgb,
        'pixel_coords': pixel_coords,
        'resolution_w': resolution_w,
        'resolution_h': resolution_h,
        'camera_idx': camera_idx,
        'frame_idx': frame_idx,
    }


class CUDAAutograd(torch.autograd.Function):
    """Autograd wrapper for CUDA PPISP."""

    @staticmethod
    def forward(ctx, exposure_params, vignetting_params,
                color_params, crf_params, rgb_in, pixel_coords,
                resolution_w, resolution_h, camera_idx, frame_idx):

        # Ensure contiguous float32
        exposure_params = exposure_params.float().contiguous()
        vignetting_params = vignetting_params.float().contiguous()
        color_params = color_params.float().contiguous()
        crf_params = crf_params.float().contiguous()
        rgb_in = rgb_in.float().contiguous()
        pixel_coords = pixel_coords.float().contiguous()

        rgb_out = ppisp_cuda.ppisp_forward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, pixel_coords,
            resolution_w, resolution_h, camera_idx, frame_idx,
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

        grads = ppisp_cuda.ppisp_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, pixel_coords,
            v_rgb_out.contiguous(),
            ctx.resolution_w, ctx.resolution_h,
            ctx.camera_idx, ctx.frame_idx,
        )

        return grads + (None,) * 5  # None for pixel_coords + 4 int args


def run_cuda_forward(params, inputs, rgb):
    """Run CUDA forward pass."""
    return CUDAAutograd.apply(
        params['exposure_params'],
        params['vignetting_params'],
        params['color_params'],
        params['crf_params'],
        rgb,
        inputs['pixel_coords'],
        inputs['resolution_w'],
        inputs['resolution_h'],
        inputs['camera_idx'],
        inputs['frame_idx'],
    )


def run_torch_forward(params, inputs, rgb):
    """Run PyTorch forward pass."""
    return ppisp_apply_torch(
        params['exposure_params'],
        params['vignetting_params'],
        params['color_params'],
        params['crf_params'],
        rgb,
        inputs['pixel_coords'],
        inputs['resolution_w'],
        inputs['resolution_h'],
        inputs['camera_idx'],
        inputs['frame_idx'],
    )


def compute_relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute relative error between two tensors."""
    abs_diff = torch.abs(a - b)
    abs_max = torch.maximum(torch.abs(a), torch.abs(b))
    rel_error = abs_diff / (abs_max + eps)
    return rel_error.max().item()


# =============================================================================
# Forward Pass Tests
# =============================================================================

def test_forward_basic():
    """Test basic forward pass equivalence."""
    params = create_test_params(num_cameras=2, num_frames=5, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=2, num_frames=5, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Forward pass max diff: {max_diff}"


def test_forward_multiple_iterations():
    """Test forward pass with multiple random parameter configurations."""
    num_iterations = 10
    atol = 1e-5  # Relaxed tolerance for float32 precision

    for iteration in range(num_iterations):
        params = create_test_params(
            num_cameras=2, num_frames=5, seed=42 + iteration)
        inputs = create_test_inputs(
            batch_size=256, num_cameras=2, num_frames=5, seed=100 + iteration)

        rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
        rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

        max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
        assert max_diff <= atol, f"Iteration {iteration}: max_diff={max_diff}"


def test_forward_different_batch_sizes():
    """Test forward pass with different batch sizes."""
    for batch_size in [1, 16, 128, 512, 1024]:
        params = create_test_params(seed=42)
        inputs = create_test_inputs(batch_size=batch_size, seed=42)

        rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
        rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

        max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
        assert max_diff < 3e-6, f"Batch size {batch_size}: max_diff={max_diff}"


def test_forward_identity_params():
    """Test forward pass with identity parameters (zeros)."""
    params = create_test_params(seed=42)

    # Set all params to zero (identity transform)
    with torch.no_grad():
        params['exposure_params'].zero_()
        params['vignetting_params'].zero_()
        params['color_params'].zero_()
        params['crf_params'].zero_()

    inputs = create_test_inputs(batch_size=256, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Identity params max_diff={max_diff}"


def test_forward_no_camera_effects():
    """Test forward pass with camera_idx=-1 (disabled camera effects)."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['camera_idx'] = -1

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"No camera effects max_diff={max_diff}"


def test_forward_no_frame_effects():
    """Test forward pass with frame_idx=-1 (disabled frame effects)."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['frame_idx'] = -1

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"No frame effects max_diff={max_diff}"


# =============================================================================
# Backward Pass Tests
# =============================================================================

def test_backward_basic():
    """Test backward pass gradient equivalence between CUDA and PyTorch.

    Compares all gradient outputs (rgb_in, exposure, vignetting, color, crf)
    between CUDA backward pass and PyTorch autograd reference.
    """
    params_cuda = create_test_params(num_cameras=2, num_frames=5, seed=42)
    params_torch = create_test_params(num_cameras=2, num_frames=5, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=2, num_frames=5, seed=42)

    # Clone inputs for both passes
    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    # Forward passes
    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    # Identical gradient for backward
    grad_output = torch.randn_like(output_cuda)

    # Backward passes
    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # Check each gradient with detailed error info
    grad_pairs = [
        ('rgb_in', rgb_cuda.grad, rgb_torch.grad),
        ('exposure', params_cuda['exposure_params'].grad,
         params_torch['exposure_params'].grad),
        ('vignetting', params_cuda['vignetting_params'].grad,
         params_torch['vignetting_params'].grad),
        ('color', params_cuda['color_params'].grad,
         params_torch['color_params'].grad),
        ('crf', params_cuda['crf_params'].grad,
         params_torch['crf_params'].grad),
    ]

    for name, grad_cuda, grad_torch in grad_pairs:
        max_diff = torch.abs(grad_cuda - grad_torch).max().item()
        rel_error = compute_relative_error(grad_cuda, grad_torch)
        assert max_diff <= atol, \
            f"{name} grad: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}"


def test_backward_no_camera_effects():
    """Test backward pass with camera_idx=-1 (disabled camera effects)."""
    params_cuda = create_test_params(seed=42)
    params_torch = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['camera_idx'] = -1

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    grad_output = torch.randn_like(output_cuda)

    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # RGB and per-frame params should have gradients
    max_diff = torch.abs(rgb_cuda.grad - rgb_torch.grad).max().item()
    assert max_diff <= atol, f"RGB grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['exposure_params'].grad -
        params_torch['exposure_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Exposure grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['color_params'].grad -
        params_torch['color_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Color grad max_diff={max_diff}"


def test_backward_no_frame_effects():
    """Test backward pass with frame_idx=-1."""
    params_cuda = create_test_params(seed=42)
    params_torch = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['frame_idx'] = -1

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    grad_output = torch.randn_like(output_cuda)

    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # RGB and per-camera params should have gradients
    max_diff = torch.abs(rgb_cuda.grad - rgb_torch.grad).max().item()
    assert max_diff <= atol, f"RGB grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['vignetting_params'].grad -
        params_torch['vignetting_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Vignetting grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['crf_params'].grad -
        params_torch['crf_params'].grad
    ).max().item()
    assert max_diff <= atol, f"CRF grad max_diff={max_diff}"


def test_backward_multiple_pixels_per_thread():
    """Backward with enough pixels that each thread accumulates several pixels.

    The CUDA backward kernel caps its grid at the resident block count and
    grid-strides over the remaining pixels, reducing the parameter gradients
    once per block; small batches never exercise that accumulation. A 960x540
    image is many strides long on any current GPU, and on most SM counts the
    stride does not divide it, so the tail is ragged as well.
    """
    # Seed chosen so the sampled vignetting falloff is not clamped, which
    # would zero the vignetting gradient and make its comparison vacuous.
    params_cuda = create_test_params(num_cameras=2, num_frames=5, seed=1)
    params_torch = create_test_params(num_cameras=2, num_frames=5, seed=1)
    inputs = create_test_inputs(
        batch_size=960 * 540, num_cameras=2, num_frames=5, seed=1)
    # Keep every pixel below CRF saturation so this test isolates the
    # reduction; the CRF derivative is ill-conditioned next to the clamp and
    # saturated pixels are covered by test_backward_saturated_pixels.
    inputs['rgb'] = inputs['rgb'] * 0.75

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    grad_output = torch.randn_like(output_cuda)
    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    grad_pairs = [
        ('rgb_in', rgb_cuda.grad, rgb_torch.grad),
        ('exposure', params_cuda['exposure_params'].grad,
         params_torch['exposure_params'].grad),
        ('vignetting', params_cuda['vignetting_params'].grad,
         params_torch['vignetting_params'].grad),
        ('color', params_cuda['color_params'].grad,
         params_torch['color_params'].grad),
        ('crf', params_cuda['crf_params'].grad,
         params_torch['crf_params'].grad),
    ]

    # Parameter gradients are float32 sums over thousands of signed terms, so
    # compare norm-relative error with a tolerance above the cancellation noise.
    # A lost pixel group would show up as a relative error of order 1.
    for name, grad_cuda, grad_torch in grad_pairs:
        ref_norm = torch.norm(grad_torch).item()
        assert ref_norm > 0, f"{name} reference grad is zero; test would be vacuous"
        rel_error = (torch.norm(grad_cuda - grad_torch).item() / ref_norm)
        assert rel_error <= 1e-3, f"{name} grad: rel_error={rel_error:.2e}"


def test_backward_saturated_pixels():
    """Backward with many pixels clamped at the ends of the CRF.

    Both implementations cut off the normalized toe/shoulder input gradients
    at the endpoints while preserving gamma gradients for every positive y.
    The forward keeps black at 0 and saturation at 1.
    """
    params_cuda = create_test_params(num_cameras=2, num_frames=5, seed=1)
    params_torch = create_test_params(num_cameras=2, num_frames=5, seed=1)
    inputs = create_test_inputs(
        batch_size=65536, num_cameras=2, num_frames=5, seed=1)
    # Push a large fraction of the channels past 1.0 before the CRF, and make
    # the first rows pure black.
    num_black = 256
    inputs['rgb'] = inputs['rgb'] * 1.3
    inputs['rgb'][:num_black] = 0.0

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)
    min_saturated_fraction = 0.1  # enough clamped channels for a meaningful test
    saturated = (output_torch >= 0.9999).sum().item()
    assert saturated > output_torch.numel() * min_saturated_fraction, (
        f"only {saturated} saturated channels; test would be vacuous")
    assert torch.equal(output_cuda[:num_black], torch.zeros_like(output_cuda[:num_black])), (
        "black input must stay exactly black")
    assert torch.equal(output_torch[:num_black], torch.zeros_like(output_torch[:num_black]))

    grad_output = torch.randn_like(output_cuda)
    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    # No pixel may receive a wildly different gradient.
    max_pixel_diff = (rgb_cuda.grad - rgb_torch.grad).abs().max().item()
    assert max_pixel_diff <= 0.5, f"rgb_in grad max pixel diff={max_pixel_diff:.2e}"

    # Pixels within float error of the clamp remain ill-conditioned, so the
    # parameter sums they dominate get a looser tolerance than the per-pixel
    # rgb gradient.
    grad_pairs = [
        ('rgb_in', rgb_cuda.grad, rgb_torch.grad, 1e-3),
        ('exposure', params_cuda['exposure_params'].grad,
         params_torch['exposure_params'].grad, 1e-2),
        ('vignetting', params_cuda['vignetting_params'].grad,
         params_torch['vignetting_params'].grad, 1e-2),
        ('color', params_cuda['color_params'].grad,
         params_torch['color_params'].grad, 1e-2),
        ('crf', params_cuda['crf_params'].grad,
         params_torch['crf_params'].grad, 1e-2),
    ]
    for name, grad_cuda, grad_torch, tol in grad_pairs:
        ref_norm = torch.norm(grad_torch).item()
        assert ref_norm > 0, f"{name} reference grad is zero; test would be vacuous"
        rel_error = (torch.norm(grad_cuda - grad_torch).item() / ref_norm)
        assert rel_error <= tol, f"{name} grad: rel_error={rel_error:.2e}"


# =============================================================================
# Stress Tests
# =============================================================================

def test_large_batch():
    """Test with large batch size."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=4096, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Large batch max_diff={max_diff}"


def test_many_cameras_frames():
    """Test with many cameras and frames."""
    num_cameras = 10
    num_frames = 50

    params = create_test_params(
        num_cameras=num_cameras, num_frames=num_frames, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=num_cameras, num_frames=num_frames, seed=42
    )

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Many cameras/frames max_diff={max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
