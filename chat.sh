#!/bin/bash
# ==============================================================================
# Interactive Chat Translation Terminal Launcher
# ==============================================================================
source /home/christian/miniforge3/etc/profile.d/conda.sh
conda activate vega_torch

# Run interactive CLI
python /git/finetune/tools/chat_cli.py "$@"
