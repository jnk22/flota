#!/usr/bin/env bash
#
# Total runs: 234

CUDA_DEVICE=${1:-0}

MODELS_EN=(bert-base-uncased xlnet-base-cased)
MODELS_DE=(bert-base-german-dbmdz-cased)
DATASETS_EN=(arxiv_cs_1e+02 arxiv_maths_1e+02 arxiv_physics_1e+02 arxiv_cs_1e+03 arxiv_maths_1e+03 arxiv_physics_1e+03)
DATASETS_DE=(10kgnad_limited)
K=(0 2 3 4)

# FLOTA-DP: English => 2x6 = 12
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    flota run --mode flota-dp --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# FLOTA-DP: German => 1
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    flota run --mode flota-dp --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# FLOTA: English => 2x6x4 = 48
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: German => 4
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# BASE: English => 2x6 => 12
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    flota run --mode base --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# BASE: German => 1
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    flota run --mode base --random-seed 123 --epochs 25 --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
  done
done

# FLOTA: English (Prefixes) => 2x6x4 = 48
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --prefixes vocabs/prefixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: German (Prefixes) => 4
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --prefixes vocabs/prefixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: English (Suffixes) => 2x6x4 = 48
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --suffixes vocabs/suffixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: German (Suffixes) => 4
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --suffixes vocabs/suffixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: English (Prefixes+Suffixes) => 2x6x4 = 48
for model in "${MODELS_EN[@]}"; do
  for data in "${DATASETS_EN[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --prefixes vocabs/prefixes_en.txt --suffixes vocabs/suffixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done

# FLOTA: German (Prefixes+Suffixes) => 4
for model in "${MODELS_DE[@]}"; do
  for data in "${DATASETS_DE[@]}"; do
    for k in "${K[@]}"; do
      flota run --mode flota --k "$k" --random-seed 123 --epochs 25 --prefixes vocabs/prefixes_en.txt --suffixes vocabs/suffixes_en.txt --cuda-device "$CUDA_DEVICE" "$model" "data/$data"
    done
  done
done
