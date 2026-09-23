/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

constexpr int PPISP_BLOCK_SIZE = 256;
constexpr int PPISP_COLOR_PARAMS = 8;
constexpr int PPISP_FRAME_MEAN_SUMS_SIZE = 1 + PPISP_COLOR_PARAMS;
constexpr int PPISP_CRF_PARAMS_PER_CHANNEL = 4;
constexpr int PPISP_VIGNETTING_PARAMS_PER_CHANNEL = 5;

// Dead zone at the endpoints of the normalized CRF toe/shoulder inputs: the
// backward zeroes the gradients through the power's base and exponent when the
// base is at or below this value. Exposed to Python as
// ppisp_cuda.CRF_BASE_GRAD_EPS so the test reference can assert it uses the same.
constexpr float PPISP_CRF_BASE_GRAD_EPS = 1e-6f;
