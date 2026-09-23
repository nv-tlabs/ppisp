# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the CUDA implementation of PPISP regularization loss."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

import ppisp
import ppisp_cuda


def _regularization_loss_torch_from_tensors(
    exposure_params: torch.Tensor,
    vignetting_params: torch.Tensor,
    color_params: torch.Tensor,
    crf_params: torch.Tensor,
    color_pinv_block_diag: torch.Tensor,
    cfg: ppisp.PPISPConfig,
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=exposure_params.device)

    if cfg.exposure_mean > 0:
        exposure_residual = exposure_params.mean()
        total_loss = total_loss + cfg.exposure_mean * F.smooth_l1_loss(
            exposure_residual, torch.zeros_like(exposure_residual), beta=0.1
        )

    if cfg.vig_center > 0:
        vig_optical_center = vignetting_params[:, :, :2]
        vig_center = (vig_optical_center ** 2).sum(dim=-1)
        total_loss = total_loss + cfg.vig_center * vig_center.mean()

    if cfg.vig_non_pos > 0:
        vig_alphas = vignetting_params[:, :, 2:]
        total_loss = total_loss + cfg.vig_non_pos * F.relu(vig_alphas).mean()

    if cfg.vig_channel > 0:
        total_loss = total_loss + cfg.vig_channel * vignetting_params.var(
            dim=1, unbiased=False
        ).mean()

    if cfg.color_mean > 0:
        color_offsets = color_params @ color_pinv_block_diag
        color_residual = color_offsets.mean(dim=0)
        total_loss = total_loss + cfg.color_mean * F.smooth_l1_loss(
            color_residual,
            torch.zeros_like(color_residual),
            beta=0.005,
            reduction="mean",
        )

    if cfg.crf_channel > 0:
        total_loss = total_loss + cfg.crf_channel * crf_params.var(
            dim=1, unbiased=False
        ).mean()

    return total_loss


def _regularization_loss_torch(module: ppisp.PPISP) -> torch.Tensor:
    return _regularization_loss_torch_from_tensors(
        module.exposure_params,
        module.vignetting_params,
        module.color_params,
        module.crf_params,
        module.color_pinv_block_diag,
        module.config,
    )


def _make_config(**overrides) -> ppisp.PPISPConfig:
    cfg = ppisp.PPISPConfig(
        use_controller=False,
        exposure_mean=0.7,
        vig_center=0.03,
        vig_channel=0.2,
        vig_non_pos=0.05,
        color_mean=1.3,
        crf_channel=0.17,
    )
    return replace(cfg, **overrides)


def _make_module(
    seed: int = 0,
    num_cameras: int = 4,
    num_frames: int = 9,
    config: ppisp.PPISPConfig | None = None,
) -> ppisp.PPISP:
    cfg = config or _make_config()
    module = ppisp.PPISP(num_cameras=num_cameras, num_frames=num_frames, config=cfg)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    with torch.no_grad():
        module.exposure_params.copy_(
            torch.randn(num_frames, device="cuda", generator=generator) * 0.2
        )
        module.vignetting_params.copy_(
            torch.randn(
                num_cameras, 3, ppisp.VIGNETTING_PARAMS_PER_CHANNEL,
                device="cuda", generator=generator,
            ) * 0.1
        )
        module.color_params.copy_(
            torch.randn(
                num_frames, ppisp.COLOR_PARAMS_PER_FRAME,
                device="cuda", generator=generator,
            ) * 0.2
        )
        module.crf_params.copy_(
            torch.randn(
                num_cameras, 3, ppisp.CRF_PARAMS_PER_CHANNEL,
                device="cuda", generator=generator,
            ) * 0.15
        )
    return module


def _weights(cfg: ppisp.PPISPConfig) -> tuple[float, float, float, float, float, float]:
    return (
        cfg.exposure_mean,
        cfg.vig_center,
        cfg.vig_channel,
        cfg.vig_non_pos,
        cfg.color_mean,
        cfg.crf_channel,
    )


def _clone_params(src: ppisp.PPISP, dst: ppisp.PPISP) -> None:
    with torch.no_grad():
        dst.exposure_params.copy_(src.exposure_params)
        dst.vignetting_params.copy_(src.vignetting_params)
        dst.color_params.copy_(src.color_params)
        dst.crf_params.copy_(src.crf_params)


def _assert_loss_and_grads_match(
    module_cuda: ppisp.PPISP,
    module_torch: ppisp.PPISP,
    loss_atol: float = 1e-6,
    grad_atol: float = 5e-6,
    grad_rtol: float = 5e-5,
) -> None:
    loss_cuda = module_cuda.get_regularization_loss()
    loss_torch = _regularization_loss_torch(module_torch)

    assert torch.allclose(loss_cuda, loss_torch, atol=loss_atol), (
        f"loss_cuda={loss_cuda.item()}, loss_torch={loss_torch.item()}"
    )

    loss_cuda.backward()
    loss_torch.backward()

    for name in ("exposure_params", "vignetting_params", "color_params", "crf_params"):
        param_cuda = getattr(module_cuda, name)
        param_torch = getattr(module_torch, name)
        grad_cuda = param_cuda.grad
        grad_torch = param_torch.grad
        if grad_cuda is None:
            grad_cuda = torch.zeros_like(param_cuda)
        if grad_torch is None:
            grad_torch = torch.zeros_like(param_torch)
        max_diff = (grad_cuda - grad_torch).abs().max().item() if grad_cuda.numel() else 0.0
        assert torch.allclose(grad_cuda, grad_torch, atol=grad_atol, rtol=grad_rtol), (
            f"{name} grad max_diff={max_diff}"
        )


def test_regularization_loss_from_state_dict_matches_torch_reference():
    cfg = _make_config()
    module_cuda = ppisp.PPISP(num_cameras=4, num_frames=9, config=cfg)
    module_torch = ppisp.PPISP(num_cameras=4, num_frames=9, config=cfg)
    _clone_params(_make_module(seed=5, config=cfg), module_cuda)

    restored = ppisp.PPISP.from_state_dict(module_cuda.state_dict(), config=cfg)

    _clone_params(module_cuda, module_torch)
    _assert_loss_and_grads_match(restored, module_torch)


def test_regularization_loss_matches_torch_reference():
    module = _make_module(seed=12)

    loss_cuda = module.get_regularization_loss()
    loss_torch = _regularization_loss_torch(module)

    assert torch.allclose(loss_cuda, loss_torch, atol=1e-6), (
        f"loss_cuda={loss_cuda.item()}, loss_torch={loss_torch.item()}"
    )


def test_regularization_loss_backward_matches_torch_reference():
    module_cuda = _make_module(seed=34)
    module_torch = _make_module(seed=35)
    _clone_params(module_cuda, module_torch)

    _assert_loss_and_grads_match(module_cuda, module_torch)


def test_regularization_loss_minimum_valid_shape_matches_torch_reference():
    module_cuda = _make_module(seed=45, num_cameras=1, num_frames=1)
    module_torch = _make_module(seed=46, num_cameras=1, num_frames=1)
    _clone_params(module_cuda, module_torch)

    _assert_loss_and_grads_match(module_cuda, module_torch)


@pytest.mark.parametrize(
    ("num_cameras", "num_frames", "seed"),
    (
        (1, 7, 101),
        (2, 1, 102),
        (5, 3, 103),
        (7, 17, 104),
        (13, 2, 105),
        (2, 129, 106),
    ),
)
def test_regularization_loss_shape_grid_matches_torch_reference(
    num_cameras: int,
    num_frames: int,
    seed: int,
):
    module_cuda = _make_module(
        seed=seed, num_cameras=num_cameras, num_frames=num_frames
    )
    module_torch = _make_module(
        seed=seed + 1000, num_cameras=num_cameras, num_frames=num_frames
    )
    _clone_params(module_cuda, module_torch)

    _assert_loss_and_grads_match(module_cuda, module_torch)


@pytest.mark.parametrize(
    "active_weight",
    ("exposure_mean", "vig_center", "vig_channel", "vig_non_pos", "color_mean", "crf_channel"),
)
def test_regularization_loss_each_term_matches_torch_reference(active_weight: str):
    weights = {
        "exposure_mean": 0.0,
        "vig_center": 0.0,
        "vig_channel": 0.0,
        "vig_non_pos": 0.0,
        "color_mean": 0.0,
        "crf_channel": 0.0,
    }
    weights[active_weight] = 1.0
    cfg = _make_config(**weights)

    module_cuda = _make_module(seed=active_weight.__hash__() % 1000, config=cfg)
    module_torch = _make_module(seed=999, config=cfg)
    _clone_params(module_cuda, module_torch)

    _assert_loss_and_grads_match(module_cuda, module_torch)


@pytest.mark.parametrize(
    ("weights", "seed"),
    (
        (
            {
                "exposure_mean": 0.0,
                "vig_center": 0.03,
                "vig_channel": 0.0,
                "vig_non_pos": 0.05,
                "color_mean": 0.0,
                "crf_channel": 0.17,
            },
            201,
        ),
        (
            {
                "exposure_mean": 0.7,
                "vig_center": 0.0,
                "vig_channel": 0.2,
                "vig_non_pos": 0.0,
                "color_mean": 1.3,
                "crf_channel": 0.0,
            },
            202,
        ),
        (
            {
                "exposure_mean": 2.5,
                "vig_center": 0.11,
                "vig_channel": 0.013,
                "vig_non_pos": 0.7,
                "color_mean": 0.05,
                "crf_channel": 3.0,
            },
            203,
        ),
        pytest.param(
            {
                "exposure_mean": 1.0,
                "vig_center": 0.02,
                "vig_channel": 0.1,
                "vig_non_pos": 0.01,
                "color_mean": 1.0,
                "crf_channel": 0.1,
            },
            204,
            id="nre_ppisp_post_processing_defaults",
        ),
        pytest.param(
            {
                "exposure_mean": 0.001,
                "vig_center": 0.02,
                "vig_channel": 0.01,
                "vig_non_pos": 0.01,
                "color_mean": 0.01,
                "crf_channel": 0.01,
            },
            205,
            id="nre_gaussians_av_ppisp_lambdas",
        ),
    ),
)
def test_regularization_loss_mixed_weights_match_torch_reference(
    weights: dict[str, float],
    seed: int,
):
    cfg = _make_config(**weights)
    module_cuda = _make_module(seed=seed, config=cfg)
    module_torch = _make_module(seed=seed + 1000, config=cfg)
    _clone_params(module_cuda, module_torch)

    _assert_loss_and_grads_match(module_cuda, module_torch)


def test_regularization_loss_zero_weights_has_zero_grads():
    cfg = ppisp.PPISPConfig(
        use_controller=False,
        exposure_mean=0.0,
        vig_center=0.0,
        vig_channel=0.0,
        vig_non_pos=0.0,
        color_mean=0.0,
        crf_channel=0.0,
    )
    module = ppisp.PPISP(num_cameras=2, num_frames=3, config=cfg)

    loss = module.get_regularization_loss()
    loss.backward()

    assert loss.item() == 0.0
    for name in ("exposure_params", "vignetting_params", "color_params", "crf_params"):
        grad = getattr(module, name).grad
        assert grad is not None
        assert torch.count_nonzero(grad).item() == 0


def test_regularization_direct_binding_forward_shape_and_value():
    module = _make_module(seed=56)

    loss_direct, stats = ppisp_cuda.ppisp_regularization_forward(
        module.exposure_params,
        module.vignetting_params,
        module.color_params,
        module.crf_params,
        *_weights(module.config),
    )
    loss_module = module.get_regularization_loss()

    assert loss_direct.shape == torch.Size([])
    assert loss_direct.device == module.exposure_params.device
    assert loss_direct.dtype == module.exposure_params.dtype
    assert stats.shape == torch.Size([1 + ppisp.COLOR_PARAMS_PER_FRAME])
    assert stats.device == module.exposure_params.device
    assert stats.dtype == module.exposure_params.dtype
    assert torch.allclose(loss_direct, loss_module, atol=1e-6)


def test_regularization_direct_binding_backward_shapes_and_grad_scale():
    module = _make_module(seed=67)
    module_torch = _make_module(seed=68)
    _clone_params(module, module_torch)

    grad_scale = torch.tensor(3.25, device="cuda")
    _, stats = ppisp_cuda.ppisp_regularization_forward(
        module.exposure_params,
        module.vignetting_params,
        module.color_params,
        module.crf_params,
        *_weights(module.config),
    )
    direct_grads = ppisp_cuda.ppisp_regularization_backward(
        module.exposure_params,
        module.vignetting_params,
        module.color_params,
        module.crf_params,
        grad_scale,
        stats,
        *_weights(module.config),
    )

    (grad_exposure, grad_vignetting, grad_color, grad_crf) = direct_grads
    expected_shapes = (
        module.exposure_params.shape,
        module.vignetting_params.shape,
        module.color_params.shape,
        module.crf_params.shape,
    )
    for grad, shape in zip(direct_grads, expected_shapes):
        assert grad.shape == shape
        assert grad.device == module.exposure_params.device
        assert grad.dtype == module.exposure_params.dtype

    (_regularization_loss_torch(module_torch) * grad_scale).backward()
    expected = (
        module_torch.exposure_params.grad,
        module_torch.vignetting_params.grad,
        module_torch.color_params.grad,
        module_torch.crf_params.grad,
    )
    for grad, ref in zip((grad_exposure, grad_vignetting, grad_color, grad_crf), expected):
        assert torch.allclose(grad, ref, atol=5e-6, rtol=5e-5)


def test_regularization_loss_boundary_values_match_torch_reference():
    cfg = _make_config()
    module_cuda = ppisp.PPISP(num_cameras=2, num_frames=4, config=cfg)
    module_torch = ppisp.PPISP(num_cameras=2, num_frames=4, config=cfg)

    with torch.no_grad():
        # Exposure mean is exactly SmoothL1 beta; vignetting alphas cover
        # negative, zero, and positive ReLU branches.
        module_cuda.exposure_params.fill_(0.1)
        module_cuda.vignetting_params.zero_()
        module_cuda.vignetting_params[:, :, 0] = 0.25
        module_cuda.vignetting_params[:, :, 1] = -0.25
        module_cuda.vignetting_params[:, :, 2:] = torch.tensor(
            [-0.2, 0.0, 0.3], device="cuda"
        )

        module_cuda.color_params.zero_()
        module_cuda.color_params[:, 0] = 0.005 / module_cuda.color_pinv_block_diag[0, 0]

        module_cuda.crf_params.zero_()
        module_cuda.crf_params[:, 0, :] = torch.tensor(
            [0.2, -0.1, 0.0, 0.3], device="cuda"
        )
        module_cuda.crf_params[:, 1, :] = torch.tensor(
            [-0.2, 0.1, 0.0, -0.3], device="cuda"
        )
        module_cuda.crf_params[:, 2, :] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], device="cuda"
        )

    _clone_params(module_cuda, module_torch)
    _assert_loss_and_grads_match(module_cuda, module_torch)


def test_regularization_loss_color_block_explicit_expected_value():
    cfg = _make_config(
        exposure_mean=0.0,
        vig_center=0.0,
        vig_channel=0.0,
        vig_non_pos=0.0,
        color_mean=1.0,
        crf_channel=0.0,
    )
    module = ppisp.PPISP(num_cameras=1, num_frames=1, config=cfg)
    with torch.no_grad():
        module.color_params.zero_()
        module.color_params[0, 7] = 1.0

    loss = module.get_regularization_loss()
    neutral_block = torch.tensor(
        [0.0128369, -0.0034654, -0.0034654, 0.0128158],
        device="cuda",
        dtype=torch.float32,
    )
    expected = (
        F.smooth_l1_loss(
            neutral_block[1],
            torch.zeros_like(neutral_block[1]),
            beta=0.005,
            reduction="sum",
        )
        + F.smooth_l1_loss(
            neutral_block[3],
            torch.zeros_like(neutral_block[3]),
            beta=0.005,
            reduction="sum",
        )
    ) / ppisp.COLOR_PARAMS_PER_FRAME

    assert torch.allclose(loss, expected, atol=1e-7)


def test_regularization_loss_channel_equal_variance_terms_are_zero():
    cfg = _make_config(
        exposure_mean=0.0,
        vig_center=0.0,
        vig_channel=1.0,
        vig_non_pos=0.0,
        color_mean=0.0,
        crf_channel=1.0,
    )
    module = ppisp.PPISP(num_cameras=3, num_frames=2, config=cfg)

    with torch.no_grad():
        vig = torch.randn(3, 1, ppisp.VIGNETTING_PARAMS_PER_CHANNEL, device="cuda")
        crf = torch.randn(3, 1, ppisp.CRF_PARAMS_PER_CHANNEL, device="cuda")
        module.vignetting_params.copy_(vig.repeat(1, 3, 1))
        module.crf_params.copy_(crf.repeat(1, 3, 1))

    loss = module.get_regularization_loss()
    loss.backward()

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-10)
    assert torch.allclose(
        module.vignetting_params.grad,
        torch.zeros_like(module.vignetting_params.grad),
        atol=1e-8,
    )
    assert torch.allclose(
        module.crf_params.grad,
        torch.zeros_like(module.crf_params.grad),
        atol=1e-8,
    )


def test_regularization_loss_large_grid_stride_reduction_matches_torch_reference():
    module_cuda = _make_module(seed=123, num_cameras=19, num_frames=513)
    module_torch = _make_module(seed=124, num_cameras=19, num_frames=513)
    _clone_params(module_cuda, module_torch)

    # The CUDA path grid-strides over the inputs and reduces with warp shuffles,
    # so the summation order differs from the PyTorch reference. This
    # intentionally uses looser tolerances than the small cases above.
    _assert_loss_and_grads_match(
        module_cuda,
        module_torch,
        loss_atol=2e-5,
        grad_atol=1e-5,
        grad_rtol=1e-4,
    )


def test_regularization_autograd_accepts_non_contiguous_inputs():
    cfg = _make_config()
    num_cameras = 3
    num_frames = 5

    exposure_base = torch.randn(num_frames, 2, device="cuda", requires_grad=True)
    vignetting_base = torch.randn(
        num_cameras, 3, ppisp.VIGNETTING_PARAMS_PER_CHANNEL * 2,
        device="cuda", requires_grad=True,
    )
    color_base = torch.randn(
        num_frames, ppisp.COLOR_PARAMS_PER_FRAME * 2,
        device="cuda", requires_grad=True,
    )
    crf_base = torch.randn(
        num_cameras, 3, ppisp.CRF_PARAMS_PER_CHANNEL * 2,
        device="cuda", requires_grad=True,
    )

    exposure = exposure_base[:, 0]
    vignetting = vignetting_base[:, :, ::2]
    color = color_base[:, ::2]
    crf = crf_base[:, :, ::2]
    assert not exposure.is_contiguous()
    assert not vignetting.is_contiguous()
    assert not color.is_contiguous()
    assert not crf.is_contiguous()

    color_pinv = ppisp._COLOR_PINV_BLOCK_DIAG.to(device="cuda")
    loss_cuda = ppisp._PPISPRegularizationFunction.apply(
        exposure,
        vignetting,
        color,
        crf,
        _weights(cfg),
    )
    loss_torch = _regularization_loss_torch_from_tensors(
        exposure,
        vignetting,
        color,
        crf,
        color_pinv,
        cfg,
    )

    assert torch.allclose(loss_cuda, loss_torch, atol=1e-6)

    loss_cuda.backward()
    for base in (exposure_base, vignetting_base, color_base, crf_base):
        assert base.grad is not None
        assert torch.isfinite(base.grad).all()


def test_as_float_contiguous_returns_ready_inputs_unchanged():
    ready = torch.zeros(4, 3, device="cuda")
    assert ppisp._as_float_contiguous(ready) is ready

    strided = torch.zeros(4, 6, device="cuda")[:, ::2]
    converted = ppisp._as_float_contiguous(strided)
    assert converted is not strided
    assert converted.is_contiguous() and converted.dtype is torch.float32
    assert torch.equal(converted, strided)

    double = torch.ones(4, 3, device="cuda", dtype=torch.float64)
    converted = ppisp._as_float_contiguous(double)
    assert converted.dtype is torch.float32
    assert torch.equal(converted, double.float())


def test_regularization_accepts_integer_weights():
    cfg = _make_config(exposure_mean=1, vig_center=0, vig_channel=0, vig_non_pos=0, color_mean=0, crf_channel=0)
    module = _make_module(seed=77, config=cfg)
    loss = module.get_regularization_loss()
    torch.testing.assert_close(loss, _regularization_loss_torch(module), atol=1e-6, rtol=1e-5)


def test_regularization_color_pinv_blocks_are_symmetric():
    color_pinv = ppisp._COLOR_PINV_BLOCK_DIAG
    for block in range(4):
        start = block * 2
        assert torch.allclose(
            color_pinv[start, start + 1],
            color_pinv[start + 1, start],
            atol=0.0,
            rtol=0.0,
        )


def _central_difference(module: ppisp.PPISP, name: str, index: tuple[int, ...]) -> float:
    param = getattr(module, name)
    eps = 1e-3
    with torch.no_grad():
        original = param[index].item()
        param[index] = original + eps
        loss_plus = module.get_regularization_loss().item()

    with torch.no_grad():
        param[index] = original - eps
        loss_minus = module.get_regularization_loss().item()

    with torch.no_grad():
        param[index] = original

    return (loss_plus - loss_minus) / (2.0 * eps)


def test_regularization_loss_tiny_finite_difference_gradients():
    module = _make_module(seed=234, num_cameras=1, num_frames=2)
    loss = module.get_regularization_loss()
    loss.backward()

    checks = (
        ("exposure_params", (0,)),
        ("vignetting_params", (0, 1, 3)),
        ("color_params", (1, 5)),
        ("crf_params", (0, 2, 1)),
    )
    for name, index in checks:
        finite_diff = _central_difference(module, name, index)
        analytic = getattr(module, name).grad[index].item()
        assert analytic == pytest.approx(finite_diff, abs=5e-3, rel=5e-2)


def test_regularization_obeys_non_default_stream():
    module = _make_module(seed=701, num_cameras=8, num_frames=257)
    reference = _make_module(seed=702, num_cameras=8, num_frames=257)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep the producer pending so a default-stream launch cannot pass by luck.
        torch.cuda._sleep(2_000_000)
        with torch.no_grad():
            module.exposure_params.add_(0.13)
            module.color_params.add_(0.02)
        loss = module.get_regularization_loss()
        (loss * 3.25).backward()
        observed = loss.clone()
        observed_grads = [p.grad.clone() for p in module.parameters()]
    stream.synchronize()
    _clone_params(module, reference)
    expected = _regularization_loss_torch(reference)
    (expected * 3.25).backward()
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=1e-5)
    for actual, parameter in zip(observed_grads, reference.parameters()):
        torch.testing.assert_close(actual, parameter.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mask", range(64))
@pytest.mark.parametrize("disabled_weight", [0.0, -0.5])
def test_regularization_forward_weight_combinations(mask, disabled_weight):
    names = ("exposure_mean", "vig_center", "vig_channel", "vig_non_pos", "color_mean", "crf_channel")
    cfg = _make_config(**{name: 0.7 if mask & (1 << i) else disabled_weight for i, name in enumerate(names)})
    module = _make_module(seed=901, num_cameras=32, num_frames=513, config=cfg)
    loss, stats = ppisp_cuda.ppisp_regularization_forward(
        module.exposure_params, module.vignetting_params, module.color_params,
        module.crf_params, *_weights(cfg),
    )
    expected = _regularization_loss_torch(module)
    torch.testing.assert_close(loss, expected, atol=2e-5, rtol=1e-5)
    expected_stats = torch.zeros_like(stats)
    if cfg.exposure_mean > 0:
        expected_stats[0] = module.exposure_params.sum()
    if cfg.color_mean > 0:
        expected_stats[1:] = (module.color_params @ module.color_pinv_block_diag).sum(dim=0)
    torch.testing.assert_close(stats, expected_stats, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("frames,cameras", [(0, 0), (0, 8), (257, 0), (255, 8), (256, 8), (257, 8)])
def test_regularization_empty_and_boundary(frames, cameras):
    """Forward and backward with empty or block-boundary frame and camera counts.

    The fused kernels replaced the old per-group early returns with a shared
    work bound and an inv_frames guard, so the backward runs here as well.
    """
    module = _make_module(seed=911, num_cameras=cameras, num_frames=frames)
    cfg = module.config
    if frames == 0:
        cfg = replace(cfg, exposure_mean=0.0, color_mean=0.0)
    if cameras == 0:
        cfg = replace(cfg, vig_center=0.0, vig_channel=0.0, vig_non_pos=0.0, crf_channel=0.0)
    if frames == 0 and cameras == 0:
        # No parameter has elements, so the reference loss is a constant without
        # a graph; check the CUDA path alone.
        loss = module.get_regularization_loss()
        assert loss.item() == 0.0
        loss.backward()
        for parameter in module.parameters():
            assert parameter.grad is not None and parameter.grad.numel() == 0
        return
    reference = _make_module(seed=911, num_cameras=cameras, num_frames=frames, config=cfg)
    _assert_loss_and_grads_match(module, reference, loss_atol=2e-5, grad_atol=1e-5, grad_rtol=1e-4)
    for parameter in module.parameters():
        assert torch.isfinite(parameter.grad).all()
