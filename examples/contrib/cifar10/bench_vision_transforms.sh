#!/bin/bash

set -eux

export OMP_NUM_THREADS=4

version="vision_stable"

torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r1.log --output_path=output_vision_bench
torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r2.log --output_path=output_vision_bench
torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r3.log --output_path=output_vision_bench


version="vision_prototype"

torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r1.log --output_path=output_vision_bench --use_vision_api_v2
torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r2.log --output_path=output_vision_bench --use_vision_api_v2
torchrun --nproc_per_node=2 main.py run --backend="nccl" &> output_${version}_r3.log --output_path=output_vision_bench --use_vision_api_v2
