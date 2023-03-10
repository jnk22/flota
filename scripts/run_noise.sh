#!/usr/bin/env bash

MODEL_NAME=$1

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
  python -u flota \
    --mode base \
    --noise test \
    --random-seed 123 \
    "$MODEL_NAME" \
    "data/$data"
done

for k in 1 2 3 4; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    python -u flota \
      --mode flota \
      --noise test \
      --random-seed 123 \
      --k "$k" \
      "$MODEL_NAME" \
      "data/$data"
  done
done