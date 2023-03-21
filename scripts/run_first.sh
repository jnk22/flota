#!/usr/bin/env bash

MODEL_NAME=$1
CUDA_DEVICE=${2:-0}

K=(1 2 3 4)
DATASETS=(
  arxiv_cs_1e+02
  arxiv_maths_1e+02
  arxiv_physics_1e+02
  arxiv_cs_1e+03
  arxiv_maths_1e+03
  arxiv_physics_1e+03
)

for data in "${DATASETS[@]}"; do
  for k in "${K[@]}"; do
    flota run \
      --mode first \
      --k "$k" \
      --random-seed 123 \
      --cuda-device "$CUDA_DEVICE" \
      "$MODEL_NAME" \
      "data/$data"
  done
done
