#!/usr/bin/env bash
set -euo pipefail

PY=".venv/Scripts/python.exe"
CACHE="outputs/torch_cache"

run_one() {
  local variant="$1"
  local model="$2"
  local metadata="data/weather_with_images_${variant}.tsv"
  local out="outputs/pv_clusters_${variant}_${model}"
  local log="logs_cluster_${variant}_${model}.txt"

  echo "[$(date '+%F %T')] START variant=${variant} model=${model}" | tee "$log"
  "$PY" cluster_pv_images.py \
    --metadata "$metadata" \
    --output-dir "$out" \
    --model "$model" \
    --weights imagenet \
    --active-hours 6-18 \
    --copy-mode copy \
    --torch-cache-dir "$CACHE" \
    --batch-size 32 \
    --overwrite >> "$log" 2>&1
  echo "[$(date '+%F %T')] DONE variant=${variant} model=${model}" | tee -a "$log"
}

for variant in 512 original; do
  for model in resnet18 resnet50 vgg16; do
    run_one "$variant" "$model"
  done
done
