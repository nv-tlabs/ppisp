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

"""Pin full float32 precision for the PyTorch reference implementation.

NGC PyTorch containers set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which runs
float32 matmuls in TF32 and drifts the reference about 1e-3 relative from
the CUDA kernels under test. Set the environment before torch initializes
so the comparison is float32 against float32 everywhere.
"""

import os

os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
