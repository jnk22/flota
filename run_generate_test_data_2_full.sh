#!/bin/bash

for model in bert-base-cased bert-base-uncased gpt2 xlnet-base-cased; do
  for mode in flota first longest; do
    for k in 1 2 3 4; do
      for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02; do
        python3 -u src/generate_test_data.py \
          --batch_size 64 \
          --model "$model" \
          --mode "$mode" \
          --k "$k" \
          --data "$data" \
          --output_dir "test_data/test_data_2_full"
      done
    done
  done
done

for model in bert-base-cased bert-base-uncased; do
  for mode in flota first longest; do
    for k in 1 2 3 4; do
      for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02; do
        python3 -u src/generate_test_data.py \
          --batch_size 64 \
          --model "$model" \
          --mode "$mode" \
          --k "$k" \
          --strict \
          --data "$data" \
          --output_dir "test_data/test_data_2_full"
      done
    done
  done
done
