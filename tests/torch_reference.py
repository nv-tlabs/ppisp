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
PyTorch reference implementation of PPISP pipeline.

This is a minimal, unoptimized port of the CUDA kernel for testing purposes:
- Verifying CUDA kernel outputs against a readable reference
- Gradient verification via torch.autograd
- Prototyping changes before porting to CUDA

The CUDA implementation is the authoritative source of truth for production.
"""

import torch
import torch.nn.functional as F

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
# Stored as 8x8 block-diagonal matrix for efficient single-matmul application
_COLOR_PINV_BLOCK_DIAG = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),  # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),  # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),  # Green
    torch.tensor([[0.0128369, -0.0034654],
                 [-0.0034654, 0.0128158]]),  # Neutral
).to(torch.float32)


class _PowDeadZone(torch.autograd.Function):
    """``base ** exponent`` whose gradients are zero for ``base <= EPS``.

    Used only for the normalized toe/shoulder bases, matching the CUDA CRF
    backward. The exact forward keeps black and saturation unchanged while
    cutting off the singular slope at the endpoints. Gamma must retain its
    gradients for every positive intermediate value.
    """

    # PPISP_CRF_BASE_GRAD_EPS, exported as ppisp_cuda.CRF_BASE_GRAD_EPS and
    # asserted equal by test_reference_dead_zone_matches_cuda_constant.
    EPS = 1e-6

    @staticmethod
    def forward(ctx, base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        out = torch.pow(base, exponent)
        ctx.save_for_backward(base, exponent, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        base, exponent, out = ctx.saved_tensors
        active = base > _PowDeadZone.EPS
        # Evaluate the singular factors only where they are used.
        safe_base = torch.where(active, base, torch.ones_like(base))
        zero = torch.zeros_like(base)
        grad_base = torch.where(
            active, grad_out * exponent * torch.pow(safe_base, exponent - 1.0), zero)
        grad_exponent = torch.where(active, grad_out * out * torch.log(safe_base), zero)
        return grad_base, grad_exponent.sum_to_size(exponent.shape)


def _get_homography_torch(color_params: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Compute color correction homography matrix from latent params.

    Minimal PyTorch port of get_homography() from ppisp.slang.
    """
    cp = color_params[frame_idx]  # [8]

    # Map latent -> real offsets via ZCA block-diagonal matrix
    block_diag = _COLOR_PINV_BLOCK_DIAG.to(cp.device)
    offsets = cp @ block_diag  # [8]

    # Extract real RG offsets for each control chromaticity
    bd = offsets[0:2]  # blue
    rd = offsets[2:4]  # red
    gd = offsets[4:6]  # green
    nd = offsets[6:8]  # neutral/gray

    # Fixed source chromaticities (r, g, 1)
    s_b = torch.tensor([0.0, 0.0, 1.0], device=cp.device)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=cp.device)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=cp.device)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0], device=cp.device)

    # Target = source + offset
    t_b = torch.stack([s_b[0] + bd[0], s_b[1] + bd[1], torch.ones_like(bd[0])])
    t_r = torch.stack([s_r[0] + rd[0], s_r[1] + rd[1], torch.ones_like(rd[0])])
    t_g = torch.stack([s_g[0] + gd[0], s_g[1] + gd[1], torch.ones_like(gd[0])])
    t_gray = torch.stack(
        [s_gray[0] + nd[0], s_gray[1] + nd[1], torch.ones_like(nd[0])])

    # T = [t_b, t_r, t_g] as columns
    T = torch.stack([t_b, t_r, t_g], dim=1)  # [3, 3]

    # Skew-symmetric matrix of t_gray
    skew = torch.stack([
        torch.stack([torch.zeros_like(t_gray[0]), -t_gray[2], t_gray[1]]),
        torch.stack([t_gray[2], torch.zeros_like(t_gray[0]), -t_gray[0]]),
        torch.stack([-t_gray[1], t_gray[0], torch.zeros_like(t_gray[0])]),
    ])  # [3, 3]

    # M = skew @ T
    M = skew @ T

    # Nullspace vector lambda via cross of two independent rows.
    # Compute all three cross products and select the one with largest magnitude
    # to avoid data-dependent control flow (breaks torch.compile).
    r0, r1, r2 = M[0], M[1], M[2]
    lam01 = torch.linalg.cross(r0, r1)
    lam02 = torch.linalg.cross(r0, r2)
    lam12 = torch.linalg.cross(r1, r2)

    n01 = (lam01 * lam01).sum()
    n02 = (lam02 * lam02).sum()
    n12 = (lam12 * lam12).sum()

    # Select cross product with largest magnitude (no branching)
    lam = torch.where(n01 >= n02,
                      torch.where(n01 >= n12, lam01, lam12),
                      torch.where(n02 >= n12, lam02, lam12))

    # Precomputed inverse of S = [s_b s_r s_g]
    S_inv = torch.tensor([
        [-1.0, -1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], device=cp.device)

    # H = T @ diag(lambda) @ S_inv
    D = torch.diag(lam)
    H = T @ D @ S_inv

    # Normalize so H[2,2] ~ 1 (using eps to avoid branching; intensity is
    # renormalized after applying H anyway)
    H = H / (H[2, 2] + 1e-10)

    return H


def ppisp_apply_torch(
    exposure_params: torch.Tensor,
    vignetting_params: torch.Tensor,
    color_params: torch.Tensor,
    crf_params: torch.Tensor,
    rgb_in: torch.Tensor,
    pixel_coords: torch.Tensor,
    resolution_w: int,
    resolution_h: int,
    camera_idx: int,
    frame_idx: int,
) -> torch.Tensor:
    """PyTorch reference implementation of PPISP pipeline.

    This is a minimal, unoptimized port of the CUDA kernel for testing.
    Use the CUDA kernel for production.

    Args:
        exposure_params: Per-frame exposure [num_frames]
        vignetting_params: Per-camera vignetting [num_cameras, 3, 5]
        color_params: Per-frame color correction [num_frames, 8]
        crf_params: Per-camera CRF [num_cameras, 3, 4]
        rgb_in: Input RGB [N, 3]
        pixel_coords: Pixel coordinates [N, 2]
        resolution_w: Image width
        resolution_h: Image height
        camera_idx: Camera index (-1 to disable per-camera effects)
        frame_idx: Frame index (-1 to disable per-frame effects)

    Returns:
        Processed RGB [N, 3]
    """
    rgb = rgb_in.clone()

    # --- Exposure compensation ---
    if frame_idx != -1:
        rgb = rgb * \
            torch.pow(torch.tensor(2.0, device=rgb.device),
                      exposure_params[frame_idx])

    # --- Vignetting ---
    def compute_vignetting_falloff(vig_params: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Compute vignetting falloff from params [5] and UV coords [N, 2]."""
        optical_center = vig_params[:2]
        alphas = vig_params[2:]
        delta = uv - optical_center.unsqueeze(0)
        r2 = (delta * delta).sum(dim=-1)
        falloff = torch.ones_like(r2)
        r2_pow = r2.clone()
        for alpha in alphas:
            falloff = falloff + alpha * r2_pow
            r2_pow = r2_pow * r2
        return falloff.clamp(0.0, 1.0)

    if camera_idx != -1:
        res_f = torch.tensor([resolution_w, resolution_h],
                             device=rgb.device, dtype=rgb.dtype)
        uv = (pixel_coords - res_f * 0.5) / res_f.max()

        # Per-channel vignetting [num_cameras, 3, 5]
        rgb_channels = []
        for ch in range(3):
            falloff = compute_vignetting_falloff(
                vignetting_params[camera_idx, ch], uv)
            rgb_channels.append(rgb[:, ch] * falloff)

        rgb = torch.stack(rgb_channels, dim=-1)

    # --- Color correction ---
    if frame_idx != -1:
        # Homography-based color correction
        H = _get_homography_torch(color_params, frame_idx)  # [3, 3]

        # RGB -> RGI (Red, Green, Intensity)
        intensity = rgb.sum(dim=-1, keepdim=True)  # [N, 1]
        rgi = torch.cat([rgb[:, 0:1], rgb[:, 1:2], intensity],
                        dim=-1)  # [N, 3]

        # Apply homography
        rgi = (H @ rgi.T).T  # [N, 3]

        # Renormalize intensity
        rgi = rgi * (intensity / (rgi[:, 2:3] + 1e-5))

        # RGI -> RGB
        r_out = rgi[:, 0]
        g_out = rgi[:, 1]
        b_out = rgi[:, 2] - r_out - g_out
        rgb = torch.stack([r_out, g_out, b_out], dim=-1)

    # --- CRF (Camera Response Function) ---
    if camera_idx != -1:
        # Parametric toe-shoulder CRF
        rgb = rgb.clamp(0.0, 1.0)

        rgb_channels = []
        for ch in range(3):
            # [4]: toe_raw, shoulder_raw, gamma_raw, center_raw
            crf = crf_params[camera_idx, ch]

            # Decode parameters with bounded activations
            toe = 0.3 + F.softplus(crf[0])       # min 0.3
            shoulder = 0.3 + F.softplus(crf[1])  # min 0.3
            gamma = 0.1 + F.softplus(crf[2])     # min 0.1
            center = torch.sigmoid(crf[3])       # 0-1

            # Compute a, b for piecewise curve
            lerp_val = toe + center * (shoulder - toe)
            a = (shoulder * center) / lerp_val
            b = 1.0 - a

            x = rgb[:, ch]
            mask_low = x <= center

            # Exact powers with a gradient dead zone next to zero, as in the
            # CUDA backward (see _PowDeadZone).
            y_low = a * _PowDeadZone.apply(x / center, toe)
            y_high = 1.0 - b * _PowDeadZone.apply((1.0 - x) / (1.0 - center), shoulder)

            # Select based on mask
            y = torch.where(mask_low, y_low, y_high)

            # Apply gamma with ordinary autograd for every positive y. Avoid
            # evaluating the singular power derivative at exactly zero.
            positive = y > 0.0
            safe_y = torch.where(positive, y, torch.ones_like(y))
            rgb_channels.append(torch.where(positive, torch.pow(safe_y, gamma), 0.0))

        rgb = torch.stack(rgb_channels, dim=-1)

    return rgb
