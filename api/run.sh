#!/bin/bash

# Just consider this script as a simple way to use the API with a hard-coded
# config (according to the spcification of model path, openface path, etc.).
conda activate facial

video_path=$1
output_dir=$2

python main.py "$video_path" --out_dir "$output_dir" \
    --smoothing_size .5 \
    --model_path models/model_weights.pth.tar \
    --openface_path ../../OpenFace/build/bin/ \
    --benchmark_path ../../pytorch-benchmarks/ \
    --batch_size 4 --n_workers 2 --device cuda:0 \
    --log --watchdir --watch_interval 60