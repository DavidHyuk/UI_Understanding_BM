#!/bin/bash
# Script to build ExecuTorch natively on aarch64 (e.g., NVIDIA DGX Spark)
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up from scripts/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project Root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "Starting ExecuTorch build for aarch64..."

# 1. Update system and install dependencies
sudo apt-get update
sudo apt-get install -y cmake ninja-build python3-dev libtinfo-dev

# 2. Install Project requirements
if [ -f "requirements.txt" ]; then
    echo "Installing project-level requirements from $PROJECT_ROOT/requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in $PROJECT_ROOT"
fi

# 3. Clone ExecuTorch if not exists
if [ ! -d "executorch" ]; then
    echo "Cloning ExecuTorch repository..."
    git clone --recursive https://github.com/pytorch/executorch.git
fi

cd executorch

# 4. Install ExecuTorch Python requirements
echo "Installing ExecuTorch dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
./install_requirements.sh

# 5. Build ExecuTorch with XNNPACK support
echo "Building ExecuTorch with XNNPACK backend..."
rm -rf cmake-out
mkdir cmake-out
cmake -B cmake-out . \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DPYTHON_EXECUTABLE=$(which python3)

cmake --build cmake-out -j$(nproc)

# 6. Build and install the pip package (wheel)
echo "Building and installing ExecuTorch pip package..."
python3 setup.py install

echo "ExecuTorch build and installation complete!"
