#!/usr/bin/env bash
#
# Total runs: 234

CUDA_DEVICE=$1

# FLOTA-DP: English => 2x6 = 12
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    flota --mode flota-dp --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# FLOTA-DP: German => 1
flota \
  --mode flota-dp \
  --random-seed 123 \
  --epochs 25 \
  --cuda-device "$CUDA_DEVICE" \
  bert-base-german-dbmdz-cased \
  data/10kgnad_limited

# FLOTA: English => 2x6x4 = 48
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    for k in 0 2 3 4; do
      flota --mode flota --k "$k" --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: German => 4
for k in 0 2 3 4; do
  flota \
    --mode flota \
    --k "$k" \
    --random-seed 123 \
    --epochs 25 \
    --cuda-device "$CUDA_DEVICE" \
    bert-base-german-dbmdz-cased \
    data/10kgnad_limited
done

# BASE: English => 2x6 => 12
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    flota --mode base --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# BASE: German => 1
flota \
  --mode base \
  --random-seed 123 \
  --epochs 25 \
  --cuda-device "$CUDA_DEVICE" \
  bert-base-german-dbmdz-cased \
  data/10kgnad_limited

# FLOTA: English (Prefixes) => 2x6x4 = 48
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    for k in 0 2 3 4; do
      flota \
        --mode flota \
        --k "$k" \
        --random-seed 123 \
        --epochs 25 \
        --prefix-vocab vocabs/prefixes_en.txt \
        --cuda-device "$CUDA_DEVICE" \
        "$model" \
        "data/$data"
    done
  done
done

# FLOTA: English (Suffixes) => 2x6x4 = 48
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    for k in 0 2 3 4; do
      flota \
        --mode flota \
        --k "$k" \
        --random-seed 123 \
        --epochs 25 \
        --suffix-vocab vocabs/suffixes_en.txt \
        --cuda-device "$CUDA_DEVICE" \
        "$model" \
        "data/$data"
    done
  done
done

# FLOTA: English (Prefixes+Suffixes) => 2x6x4 = 48
for model in bert-base-uncased xlnet-base-cased; do
  for data in arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03; do
    for k in 0 2 3 4; do
      flota \
        --mode flota \
        --k "$k" \
        --random-seed 123 \
        --epochs 25 \
        --prefix-vocab vocabs/prefixes_en.txt \
        --suffix-vocab vocabs/suffixes_en.txt \
        --cuda-device "$CUDA_DEVICE" \
        "$model" \
        "data/$data"
    done
  done
done

# FLOTA: German (Prefixes) => 4
for k in 0 2 3 4; do
  flota \
    --mode flota \
    --k "$k" \
    --random-seed 123 \
    --epochs 25 \
    --prefix-vocab vocabs/prefixes_de.txt \
    --cuda-device "$CUDA_DEVICE" \
    bert-base-german-dbmdz-cased \
    data/10kgnad_limited
done

# FLOTA: German (Suffixes) => 4
for k in 0 2 3 4; do
  flota \
    --mode flota \
    --k "$k" \
    --random-seed 123 \
    --epochs 25 \
    --suffix-vocab vocabs/suffixes_de.txt \
    --cuda-device "$CUDA_DEVICE" \
    bert-base-german-dbmdz-cased \
    data/10kgnad_limited
done

# FLOTA: German (Prefixes+Suffixes) => 4
for k in 0 2 3 4; do
  flota \
    --mode flota \
    --k "$k" \
    --random-seed 123 \
    --epochs 25 \
    --prefix-vocab vocabs/prefixes_de.txt \
    --suffix-vocab vocabs/suffixes_de.txt \
    --cuda-device "$CUDA_DEVICE" \
    bert-base-german-dbmdz-cased \
    data/10kgnad_limited
done
