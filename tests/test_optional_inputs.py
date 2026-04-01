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

"""Tests for optional pixel_coords and resolution inputs.

Verifies that omitting pixel_coords and/or resolution produces results
identical to providing them explicitly with pixel-center coordinates.
"""

import torch
import pytest

import ppisp


def _make_pixel_centers(H: int, W: int, device: str = "cuda") -> torch.Tensor:
    """Create explicit pixel-center coordinates [H, W, 2] matching kernel default."""
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def _make_params(seed: int = 42):
    torch.manual_seed(seed)
    return {
        "exposure_params": (torch.randn(1, device="cuda") * 0.1).requires_grad_(True),
        "vignetting_params": (torch.randn(1, 3, 5, device="cuda") * 0.1).requires_grad_(True),
        "color_params": (torch.randn(1, 8, device="cuda") * 0.1).requires_grad_(True),
        "crf_params": (torch.randn(1, 3, 4, device="cuda") * 0.1).requires_grad_(True),
    }


H, W = 32, 48


def test_pixel_coords_none_matches_explicit():
    """Omitting pixel_coords should match explicit pixel-center coords."""
    params = _make_params()
    rgb = torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2

    pixel_coords = _make_pixel_centers(H, W)
    out_explicit = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=pixel_coords,
                                     resolution_w=W, resolution_h=H,
                                     camera_idx=0, frame_idx=0)

    out_none = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=None,
                                 resolution_w=W, resolution_h=H,
                                 camera_idx=0, frame_idx=0)

    assert torch.allclose(out_explicit, out_none, atol=1e-6), \
        f"max diff: {(out_explicit - out_none).abs().max().item()}"


def test_resolution_none_matches_explicit():
    """Omitting resolution should infer (W, H) from [H, W, 3] rgb."""
    module = ppisp.PPISP(num_cameras=1, num_frames=1)
    rgb = torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2
    pixel_coords = _make_pixel_centers(H, W)

    out_explicit = module(rgb, pixel_coords=pixel_coords, resolution=(W, H),
                          camera_idx=0, frame_idx=0)
    out_inferred = module(rgb, pixel_coords=pixel_coords, resolution=None,
                          camera_idx=0, frame_idx=0)

    assert torch.allclose(out_explicit, out_inferred, atol=1e-6), \
        f"max diff: {(out_explicit - out_inferred).abs().max().item()}"


def test_both_none_matches_explicit():
    """Omitting both pixel_coords and resolution should match fully explicit call."""
    module = ppisp.PPISP(num_cameras=1, num_frames=1)
    rgb = torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2
    pixel_coords = _make_pixel_centers(H, W)

    out_explicit = module(rgb, pixel_coords=pixel_coords, resolution=(W, H),
                          camera_idx=0, frame_idx=0)
    out_none = module(rgb, pixel_coords=None, resolution=None,
                      camera_idx=0, frame_idx=0)

    assert torch.allclose(out_explicit, out_none, atol=1e-6), \
        f"max diff: {(out_explicit - out_none).abs().max().item()}"


def test_pixel_coords_none_backward():
    """Gradients should match between explicit and omitted pixel_coords."""
    params_a = _make_params(seed=99)
    params_b = _make_params(seed=99)
    rgb_a = (torch.rand(H, W, 3, device="cuda")
             * 0.6 + 0.2).requires_grad_(True)
    rgb_b = rgb_a.detach().clone().requires_grad_(True)

    pixel_coords = _make_pixel_centers(H, W)

    out_a = ppisp.ppisp_apply(**params_a, rgb_in=rgb_a, pixel_coords=pixel_coords,
                              resolution_w=W, resolution_h=H,
                              camera_idx=0, frame_idx=0)
    out_b = ppisp.ppisp_apply(**params_b, rgb_in=rgb_b, pixel_coords=None,
                              resolution_w=W, resolution_h=H,
                              camera_idx=0, frame_idx=0)

    grad = torch.randn_like(out_a)
    out_a.backward(grad)
    out_b.backward(grad)

    assert torch.allclose(rgb_a.grad, rgb_b.grad, atol=1e-5), \
        f"rgb grad max diff: {(rgb_a.grad - rgb_b.grad).abs().max().item()}"

    for name in ("exposure_params", "vignetting_params", "color_params", "crf_params"):
        diff = (params_a[name].grad - params_b[name].grad).abs().max().item()
        assert diff < 1e-5, f"{name} grad max diff: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
