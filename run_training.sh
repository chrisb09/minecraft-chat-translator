#!/bin/bash
# ==============================================================================
# Vega 56 ROCm Fine-Tuning Execution Script
# ==============================================================================
set -e

# 1. Source Conda environment
source /home/christian/miniforge3/etc/profile.d/conda.sh
conda activate vega_torch

# 2. Permanent Vega 56 ROCm Architectural Overrides (from rocm_test/source.sh)
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export ROCM_TARGET_GFX_VERSION=900
export MIOPEN_DEBUG_DISABLE_AMD_ASM_KERNELS=1
unset ROCBLAS_TENSILE_LIBPATH
export HIP_VISIBLE_DEVICES=0
export ROCM_HOME=/usr
export HIP_PATH=/usr

# 3. Enable unbuffered real-time stdout
export PYTHONUNBUFFERED=1

# 4. Launch Native PyTorch Training
python /git/finetune/training/train_native.py "$@"
