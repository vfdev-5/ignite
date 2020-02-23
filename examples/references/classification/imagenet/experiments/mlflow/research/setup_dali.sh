#!/bin/bash

python -c "import nvidia.dali"
res=$?

if [ "$res" -eq "1" ]; then

    echo "Install NVIDIA DALI"
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

fi
