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

"""Independent CRF gradient checks for the reference and CUDA implementation."""

import math

import pytest
import torch

from tests.torch_reference import _PowDeadZone, ppisp_apply_torch


@pytest.fixture(params=["cpu", "cuda"])
def crf_apply(request):
    device = request.param
    backend = ppisp_apply_torch
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        from ppisp import ppisp_apply

        backend = ppisp_apply

    return _crf_only(backend, device), device


def _crf_only(backend, device):
    exposure = torch.zeros(1, device=device)
    vignetting = torch.zeros(1, 3, 5, device=device)
    color = torch.zeros(1, 8, device=device)
    coords = torch.full((1, 2), 0.5, device=device)

    def apply(rgb, crf):
        # Disable frame effects and use the optical center to isolate the CRF.
        return backend(exposure, vignetting, color, crf, rgb, coords, 1, 1, 0, -1)

    return apply


def _crf_params(toe, gamma, device):
    raw = [math.log(math.expm1(toe - 0.3)), math.log(math.expm1(1.0 - 0.3)),
           math.log(math.expm1(gamma - 0.1)), 0.0]
    return torch.tensor(raw, device=device).view(1, 1, 4).repeat(1, 3, 1).requires_grad_()


@pytest.mark.parametrize("toe,gamma,value", [(5.0, 0.2, 0.04), (2.0, 0.5, 0.0004)])
def test_crf_shadow_gradients_match_finite_differences(crf_apply, toe, gamma, value):
    apply, device = crf_apply
    crf = _crf_params(toe, gamma, device)
    rgb = torch.full((1, 3), value, device=device, requires_grad=True)

    # The toe and gamma exponents cancel: output = a**gamma * (x / center).
    # Intermediate y is below the former cutoff despite a finite, normal slope.
    a = 1.0 / (1.0 + toe)
    assert a * (value / 0.5) ** toe < 1e-6
    slope = a ** gamma / 0.5
    out = apply(rgb, crf)
    torch.testing.assert_close(out, torch.full_like(out, value * slope), rtol=1e-5, atol=0)
    out.sum().backward()
    torch.testing.assert_close(rgb.grad, torch.full_like(rgb, slope), rtol=1e-5, atol=0)

    with torch.no_grad():
        rgb_eps = value * 0.01
        rgb_fd = (apply(rgb + rgb_eps, crf) - apply(rgb - rgb_eps, crf)) / (2 * rgb_eps)
        torch.testing.assert_close(rgb.grad, rgb_fd, rtol=1e-3, atol=0)

        param_eps = 1e-3
        for index in range(4):
            plus, minus = crf.clone(), crf.clone()
            plus[:, :, index] += param_eps
            minus[:, :, index] -= param_eps
            param_fd = (apply(rgb, plus) - apply(rgb, minus)) / (2 * param_eps)
            torch.testing.assert_close(crf.grad[:, :, index], param_fd, rtol=1e-2, atol=1e-7)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_crf_endpoints_have_finite_zero_gradients(crf_apply, value):
    apply, device = crf_apply
    crf = _crf_params(5.0, 0.2, device)
    rgb = torch.full((1, 3), value, device=device, requires_grad=True)
    out = apply(rgb, crf)
    torch.testing.assert_close(out, rgb, rtol=0, atol=0)
    out.sum().backward()
    torch.testing.assert_close(rgb.grad, torch.zeros_like(rgb), rtol=0, atol=0)
    torch.testing.assert_close(crf.grad, torch.zeros_like(crf), rtol=0, atol=0)


def test_reference_dead_zone_matches_cuda_constant():
    """The reference mirrors the kernel's dead zone; the extension exports the value."""
    ppisp_cuda = pytest.importorskip("ppisp_cuda")

    as_float32 = torch.tensor(_PowDeadZone.EPS, dtype=torch.float32).item()
    assert as_float32 == ppisp_cuda.CRF_BASE_GRAD_EPS


def _exact_crf_fp64(x, toe, shoulder, gamma, center):
    """The CRF with plain float64 powers: the model without any gradient dead zone."""
    lerp = toe + center * (shoulder - toe)
    a = shoulder * center / lerp
    b = 1.0 - a
    y = torch.where(x <= center, a * torch.pow(x / center, toe),
                    1.0 - b * torch.pow((1.0 - x) / (1.0 - center), shoulder))
    return torch.pow(y.clamp(min=0.0), gamma)


@pytest.mark.parametrize("toe,gamma,value", [(0.5, 0.5, 1e-6), (3.0, 0.11, 1e-6), (0.31, 0.31, 1e-4)])
def test_crf_near_black_gradients_match_exact_model(crf_apply, toe, gamma, value):
    """Just above the dead zone with toe*gamma < 1 the slope is steep but exact."""
    apply, device = crf_apply
    crf = _crf_params(toe, gamma, device)
    rgb = torch.full((1, 3), value, device=device, requires_grad=True)
    out = apply(rgb, crf)
    out.sum().backward()

    x = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    params = [torch.tensor(v, dtype=torch.float64) for v in (toe, 1.0, gamma, 0.5)]
    ref = _exact_crf_fp64(x, *params)
    ref.sum().backward()
    torch.testing.assert_close(out.double()[0].cpu(), ref.expand(3), rtol=1e-4, atol=0)
    torch.testing.assert_close(rgb.grad.double()[0].cpu(), x.grad.expand(3), rtol=1e-3, atol=0)


def _dead_zone_cases():
    """CRF inputs whose normalized base straddles the dead zone, as (x, active)."""
    eps = torch.tensor(_PowDeadZone.EPS, dtype=torch.float32)  # CRF_BASE_GRAD_EPS
    below = torch.nextafter(eps, torch.tensor(0.0)).item()
    above = torch.nextafter(eps, torch.tensor(1.0)).item()
    # Toe end: base = x / 0.5 is exact, so the base lands on and beside EPS.
    cases = [(base * 0.5, base > eps.item()) for base in (below, eps.item(), above)]
    # Shoulder end: 1 - x steps by 2**-24 just below 1, so base = (1 - x) / 0.5
    # steps by 2**-23 and cannot equal EPS; take the neighbours on either side.
    k = math.floor(eps.item() / 2**-23)
    cases += [(1.0 - k * 2**-24, False), (1.0 - (k + 1) * 2**-24, True)]
    return cases


@pytest.mark.parametrize("value,active", _dead_zone_cases())
def test_crf_dead_zone_transition(crf_apply, value, active):
    """rgb gradients are zero up to and at EPS and match the exact model above it."""
    apply, device = crf_apply
    # toe=5, gamma=0.2 and shoulder=1 keep the exact slope finite at both endpoints.
    crf = _crf_params(5.0, 0.2, device)
    rgb = torch.full((1, 3), value, device=device, requires_grad=True)
    out = apply(rgb, crf)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(crf.grad).all()
    if not active:
        torch.testing.assert_close(rgb.grad, torch.zeros_like(rgb), rtol=0, atol=0)
        return
    x = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    params = [torch.tensor(v, dtype=torch.float64) for v in (5.0, 1.0, 0.2, 0.5)]
    _exact_crf_fp64(x, *params).sum().backward()
    assert x.grad.item() > 0
    torch.testing.assert_close(rgb.grad.double()[0].cpu(), x.grad.expand(3), rtol=1e-3, atol=0)


@pytest.mark.parametrize("value,active", _dead_zone_cases())
def test_crf_dead_zone_cuda_matches_reference(value, active):
    """Output and every gradient agree between the CUDA kernel and the reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    from ppisp import ppisp_apply

    results = []
    for device, backend in (("cpu", ppisp_apply_torch), ("cuda", ppisp_apply)):
        crf = _crf_params(5.0, 0.2, device)
        rgb = torch.full((1, 3), value, device=device, requires_grad=True)
        out = _crf_only(backend, device)(rgb, crf)
        out.sum().backward()
        results.append([t.detach().cpu() for t in (out, rgb.grad, crf.grad)])
    (ref_out, ref_rgb, ref_crf), (cuda_out, cuda_rgb, cuda_crf) = results
    torch.testing.assert_close(cuda_out, ref_out, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(cuda_rgb, ref_rgb, rtol=1e-3, atol=0)
    torch.testing.assert_close(cuda_crf, ref_crf, rtol=1e-3, atol=1e-6)
