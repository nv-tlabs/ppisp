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
Setup script for ppisp CUDA extension.
Based on https://github.com/rahul-goel/fused-ssim/blob/main/setup.py
"""

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

from setuptools import setup

_v = {}
exec(open(os.path.join(os.path.dirname(__file__), "ppisp", "_version.py")).read(), _v)


# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


def get_cuda_arch_flags():
    """
    Determine CUDA architecture flags with the following priority:
    1. TORCH_CUDA_ARCH_LIST environment variable (standard PyTorch convention)
    2. Runtime GPU detection via torch.cuda.get_device_capability()
    3. Conservative fallback defaults (no Blackwell/compute_120, requires CUDA 12.8+)
    """
    # Priority 1: Check TORCH_CUDA_ARCH_LIST environment variable
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list_env:
        # Parse format: "7.5;8.0;8.6" or "7.5 8.0 8.6"
        arch_entries = arch_list_env.replace(";", " ").split()
        gencode_flags = []
        for entry in arch_entries:
            entry = entry.strip()
            if not entry:
                continue
            # Handle entries like "8.6" or "8.6+PTX"
            has_ptx = "+PTX" in entry
            arch_version = entry.replace("+PTX", "").strip()
            if "." in arch_version:
                major, minor = arch_version.split(".")
                compute_arch = f"compute_{major}{minor}"
                sm_arch = f"sm_{major}{minor}"
            else:
                # Handle entries like "86"
                compute_arch = f"compute_{arch_version}"
                sm_arch = f"sm_{arch_version}"
            if has_ptx:
                gencode_flags.append(
                    f"-gencode=arch={compute_arch},code={compute_arch}")
            gencode_flags.append(
                f"-gencode=arch={compute_arch},code={sm_arch}")
        if gencode_flags:
            msg = f"Using TORCH_CUDA_ARCH_LIST: {arch_list_env}"
            print(msg)
            print(msg, file=sys.stderr, flush=True)
            return gencode_flags, f"TORCH_CUDA_ARCH_LIST={arch_list_env}"

    # Priority 2: Runtime GPU detection
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
            msg = f"Detected GPU architecture: {arch}"
            print(msg)
            print(msg, file=sys.stderr, flush=True)
            return [f"-arch={arch}"], arch
    except Exception as e:
        error_msg = f"Failed to detect GPU architecture: {e}"
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)

    # Priority 3: Conservative fallback defaults
    # Note: compute_120 (Blackwell) requires CUDA 12.8+, so it's excluded
    fallback_archs = [
        "-gencode=arch=compute_75,code=sm_75",   # Turing, CUDA 10+
        "-gencode=arch=compute_80,code=sm_80",   # Ampere, CUDA 11+
        "-gencode=arch=compute_86,code=sm_86",   # Ampere GA10x, CUDA 11.1+
        "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace, CUDA 11.8+
        "-gencode=arch=compute_90,code=sm_90",   # Hopper, CUDA 12.0+
    ]
    msg = "Using conservative fallback architectures (no Blackwell/compute_120)"
    print(msg)
    print(msg, file=sys.stderr, flush=True)
    return fallback_archs, "multiple architectures (fallback)"


# Get CUDA architecture flags
arch_flags, detected_arch = get_cuda_arch_flags()

nvcc_args = [
    "-O3",
    "--use_fast_math",
]
nvcc_args += ["-lineinfo", "--generate-line-info", "--source-in-ptx"]
nvcc_args.extend(arch_flags)


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "=" * 50)
        print(arch_info)
        print("=" * 50 + "\n")
        super().build_extensions()


setup(
    name="ppisp",
    version=_v["__version__"],
    packages=["ppisp"],
    ext_modules=[
        CUDAExtension(
            name="ppisp_cuda",
            sources=[
                "ppisp/src/ppisp_impl.cu",
                "ppisp/ext.cpp"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)
