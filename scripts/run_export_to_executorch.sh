#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# flatc 경로를 PATH에 추가하고 PYTHONPATH도 설정
export PATH=$PATH:$(pwd)/executorch/cmake-out/third-party/flatc_ep/bin
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/executorch

# 변환 스크립트 실행 (scripts/ 경로 포함)
python scripts/export_to_executorch.py \
  --checkpoint models/gemma-3n \
  --model_name gemma3n \
  --quantization int4
