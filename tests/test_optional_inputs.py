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


@pytest.fixture
def validate_inputs():
    """Enable the opt-in extension input checks for one test."""
    previous = ppisp.validate_inputs_enabled()
    ppisp.set_validate_inputs(True)
    yield
    ppisp.set_validate_inputs(previous)


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


# 960x540 exceeds the capped backward grid on any current GPU, so every thread
# derives several pixel centers from its grid-stride loop index.
@pytest.mark.parametrize("height,width", [(H, W), (540, 960)])
def test_pixel_coords_none_backward(height, width):
    """Gradients should match between explicit and omitted pixel_coords."""
    params_a = _make_params(seed=99)
    params_b = _make_params(seed=99)
    rgb_a = (torch.rand(height, width, 3, device="cuda")
             * 0.6 + 0.2).requires_grad_(True)
    rgb_b = rgb_a.detach().clone().requires_grad_(True)

    pixel_coords = _make_pixel_centers(height, width)

    out_a = ppisp.ppisp_apply(**params_a, rgb_in=rgb_a, pixel_coords=pixel_coords,
                              resolution_w=width, resolution_h=height,
                              camera_idx=0, frame_idx=0)
    out_b = ppisp.ppisp_apply(**params_b, rgb_in=rgb_b, pixel_coords=None,
                              resolution_w=width, resolution_h=height,
                              camera_idx=0, frame_idx=0)

    grad = torch.randn_like(out_a)
    out_a.backward(grad)
    out_b.backward(grad)

    assert torch.allclose(rgb_a.grad, rgb_b.grad, atol=1e-5), \
        f"rgb grad max diff: {(rgb_a.grad - rgb_b.grad).abs().max().item()}"

    # Parameter gradients are atomic sums over every pixel, so their block
    # order differs between the two runs; compare relative to the magnitude.
    for name in ("exposure_params", "vignetting_params", "color_params", "crf_params"):
        torch.testing.assert_close(params_a[name].grad, params_b[name].grad,
                                   rtol=1e-4, atol=1e-5, msg=lambda m: f"{name}: {m}")


def test_parameter_grads_do_not_share_storage():
    """Each parameter's .grad owns its allocation after the CUDA backward."""
    params = _make_params()
    rgb = (torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2).requires_grad_(True)
    out = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=None,
                            resolution_w=W, resolution_h=H, camera_idx=0, frame_idx=0)
    out.sum().backward()
    grads = [p.grad for p in params.values()]
    storages = {g.untyped_storage().data_ptr() for g in grads}
    assert len(storages) == len(grads), "parameter grads alias one storage"
    for g in grads:
        assert g.storage_offset() == 0
        assert g.untyped_storage().nbytes() == g.numel() * g.element_size()


def test_input_validation_is_off_by_default():
    assert not ppisp.validate_inputs_enabled()


@pytest.mark.usefixtures("validate_inputs")
def test_mixed_device_inputs_raise():
    """With validation on, inputs off the CUDA device are rejected before any kernel runs."""
    params = _make_params()
    rgb = torch.rand(H, W, 3, device="cuda")
    with pytest.raises(RuntimeError, match="'rgb_in' is on CPU, but expected it to be on GPU"):
        ppisp.ppisp_apply(**params, rgb_in=rgb.cpu(), pixel_coords=None,
                          resolution_w=W, resolution_h=H, camera_idx=0, frame_idx=0)
    params["crf_params"] = params["crf_params"].detach().cpu()
    with pytest.raises(RuntimeError, match="'crf_params' is on CPU, but expected it to be on GPU"):
        ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=None,
                          resolution_w=W, resolution_h=H, camera_idx=0, frame_idx=0)


@pytest.mark.parametrize("name,shape,message", [
    ("vignetting_params", (1, 3, 4),
     r"Expected tensor of size \[1, 3, 5\], but got tensor of size \[1, 3, 4\] for argument #2 'vignetting_params'"),
    ("vignetting_params", (1, 15),
     "Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #2 'vignetting_params'"),
    ("crf_params", (1, 4, 3),
     r"Expected tensor of size \[1, 3, 4\], but got tensor of size \[1, 4, 3\] for argument #4 'crf_params'"),
    ("color_params", (2, 8),
     r"Expected tensor of size \[1, 8\], but got tensor of size \[2, 8\] for argument #3 'color_params'"),
    ("exposure_params", (1, 1),
     "Expected 1-dimensional tensor, but got 2-dimensional tensor for argument #1 'exposure_params'"),
])
@pytest.mark.usefixtures("validate_inputs")
def test_wrong_parameter_shapes_raise(name, shape, message):
    """The kernels hard-code the parameter layouts, so validation checks them."""
    params = _make_params()
    params[name] = torch.zeros(shape, device="cuda")
    rgb = torch.rand(H, W, 3, device="cuda")
    with pytest.raises(RuntimeError, match=message):
        ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=None,
                          resolution_w=W, resolution_h=H, camera_idx=0, frame_idx=0)


@pytest.mark.usefixtures("validate_inputs")
def test_direct_binding_rejects_non_contiguous_and_wrong_dtype():
    """With validation on, callers of the extension itself get an error, not garbage pixels."""
    import ppisp_cuda

    params = {k: v.detach() for k, v in _make_params().items()}
    rgb = torch.rand(3, H * W, device="cuda").t()  # [N, 3] but not contiguous
    with pytest.raises(RuntimeError, match="non-contiguous tensor for argument #5 'rgb_in'"):
        ppisp_cuda.ppisp_forward(*params.values(), rgb, None, W, H, 0, 0)
    with pytest.raises(RuntimeError, match="argument #5 'rgb_in' to have scalar type Float"):
        ppisp_cuda.ppisp_forward(*params.values(), rgb.contiguous().double(), None, W, H, 0, 0)


def test_misaligned_color_params_view_matches_aligned():
    """A color slice at an odd float offset in a flat buffer is copied, not faulted."""
    aligned = _make_params()
    flat = torch.zeros(1 + 8, device="cuda", requires_grad=True)  # exposure then color
    with torch.no_grad():
        flat[1:].view(1, 8).copy_(aligned["color_params"])
    misaligned = dict(aligned)
    misaligned["color_params"] = flat[1:].view(1, 8)
    assert misaligned["color_params"].is_contiguous()
    assert misaligned["color_params"].data_ptr() % 8 == 4

    rgb = (torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2)
    kwargs = dict(rgb_in=rgb, pixel_coords=None, resolution_w=W, resolution_h=H,
                  camera_idx=0, frame_idx=0)
    out_aligned = ppisp.ppisp_apply(**aligned, **kwargs)
    out_misaligned = ppisp.ppisp_apply(**misaligned, **kwargs)
    torch.testing.assert_close(out_misaligned, out_aligned, rtol=0, atol=0)
    out_aligned.sum().backward()
    out_misaligned.sum().backward()
    torch.testing.assert_close(flat.grad[1:].view(1, 8), aligned["color_params"].grad,
                               rtol=1e-4, atol=1e-5)
    assert flat.grad[0] == 0


def test_misaligned_pixel_coords_view_matches_aligned():
    """A pixel_coords slice at an odd float offset in a flat buffer is copied, not faulted."""
    import ppisp_cuda

    params = _make_params()
    coords = _make_pixel_centers(H, W).view(-1, 2)
    flat = torch.zeros(1 + coords.numel(), device="cuda")
    flat[1:].view(-1, 2).copy_(coords)
    misaligned = flat[1:].view(-1, 2)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 8 == 4

    rgb = (torch.rand(H * W, 3, device="cuda") * 0.6 + 0.2).requires_grad_(True)
    kwargs = dict(resolution_w=W, resolution_h=H, camera_idx=0, frame_idx=0)
    out_aligned = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=coords, **kwargs)
    out_misaligned = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=misaligned, **kwargs)
    torch.testing.assert_close(out_misaligned, out_aligned, rtol=0, atol=0)
    # ppisp_apply aligns before the extension; the extension aligns for direct callers.
    detached = [p.detach() for p in params.values()]
    out_binding = ppisp_cuda.ppisp_forward(*detached, rgb.detach(), misaligned, W, H, 0, 0)
    torch.testing.assert_close(out_binding, out_aligned, rtol=0, atol=0)
    grads_aligned = torch.autograd.grad(out_aligned.sum(), [rgb, *params.values()])
    grads_misaligned = torch.autograd.grad(out_misaligned.sum(), [rgb, *params.values()])
    # The rgb gradient is per pixel, but the parameter gradients are atomic sums
    # whose block order differs between the two runs, as in the backward test above.
    torch.testing.assert_close(grads_misaligned[0], grads_aligned[0], rtol=0, atol=0)
    for g_misaligned, g_aligned in zip(grads_misaligned[1:], grads_aligned[1:]):
        torch.testing.assert_close(g_misaligned, g_aligned, rtol=1e-4, atol=1e-5)


def test_pixel_coords_device_ignored_when_camera_disabled():
    """camera_idx=None drops pixel_coords before the extension, so any device works."""
    params = _make_params()
    rgb = (torch.rand(H, W, 3, device="cuda") * 0.6 + 0.2).requires_grad_(True)
    coords_cpu = _make_pixel_centers(H, W, device="cpu")
    out = ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=coords_cpu,
                            resolution_w=W, resolution_h=H, camera_idx=None, frame_idx=0)
    out.sum().backward()
    assert torch.isfinite(rgb.grad).all()


@pytest.mark.parametrize("camera_idx,frame_idx", [(1, 0), (-2, 0), (0, 1), (0, -2)])
@pytest.mark.usefixtures("validate_inputs")
def test_out_of_range_indices_raise_index_error(camera_idx, frame_idx):
    """With validation on, indices outside the parameters are rejected before any kernel runs."""
    params = _make_params()
    rgb = torch.rand(H, W, 3, device="cuda")
    with pytest.raises(IndexError, match="out of range"):
        ppisp.ppisp_apply(**params, rgb_in=rgb, pixel_coords=None,
                          resolution_w=W, resolution_h=H,
                          camera_idx=camera_idx, frame_idx=frame_idx)


@pytest.mark.usefixtures("validate_inputs")
def test_regularization_binding_rejects_malformed_inputs():
    """With validation on, the regularization wrappers check parameters and saved tensors too."""
    import ppisp_cuda

    params = {k: v.detach() for k, v in _make_params().items()}
    weights = [1.0] * 6
    bad = dict(params, crf_params=torch.zeros(1, 4, 3, device="cuda"))
    with pytest.raises(RuntimeError, match="argument #4 'crf_params'"):
        ppisp_cuda.ppisp_regularization_forward(*bad.values(), *weights)
    _, frame_mean_sums = ppisp_cuda.ppisp_regularization_forward(*params.values(), *weights)
    grad_loss = torch.ones((), device="cuda")
    with pytest.raises(RuntimeError, match="'grad_loss' is on CPU"):
        ppisp_cuda.ppisp_regularization_backward(*params.values(), grad_loss.cpu(),
                                                 frame_mean_sums, *weights)
    with pytest.raises(RuntimeError, match="argument #6 'frame_mean_sums'"):
        ppisp_cuda.ppisp_regularization_backward(*params.values(), grad_loss,
                                                 frame_mean_sums[:-1], *weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
