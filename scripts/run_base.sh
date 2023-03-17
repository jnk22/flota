#!/usr/bin/env bash

MODEL_NAME=$1

for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
  flota run \
    --mode base \
    --random-seed 123 \
    "$MODEL_NAME" \
    "data/$data"
done
