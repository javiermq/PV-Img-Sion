#!/bin/bash

for day in $(seq -w 1 31); do
    echo "Procesando día $day..."

    python crop_then_project_fisheye.py \
      --input "data/sion/June-Aug/07/$day" \
      --output "data/procesadas/07/$day" \
      --cx 2000 \
      --cy 1600 \
      --r 800 \
      --projection stereographic \
      --projected_size 64 \
      --crop_size 64
done