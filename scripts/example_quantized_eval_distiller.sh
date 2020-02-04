#!/bin/bash

python3 /nnedovic.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /nnedovic.scratch.DL/ \
  --arch resnet50 \
  --resume /nnedovic.scratch.DL/checkpoints/resnet-50/precision_sweep/2018.10.10-015519/best.pth.tar \
  --evaluate \
  --workers 4 \
  --batch-size 256
