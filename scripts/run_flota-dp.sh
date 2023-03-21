#!/usr/bin/env bash

MODEL_NAME=$1
CUDA_DEVICE=${2:-0}

DATASETS=(
  arxiv_cs_1e+02
  arxiv_maths_1e+02
  arxiv_physics_1e+02
  arxiv_cs_1e+03
  arxiv_maths_1e+03
  arxiv_physics_1e+03
)

for data in "${DATASETS[@]}"; do
  flota run \
    --mode flota-dp \
    --random-seed 123 \
    --cuda-device "$CUDA_DEVICE" \
    "$MODEL_NAME" \
    "data/$data"
done
