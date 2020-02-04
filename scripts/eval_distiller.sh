#!/bin/bash

python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /arekhi.scratch.DL/checkpoints/resnet-50/tmp \
  --arch resnet50 \
  --pretrained \
  --evaluate \
  --workers 30 \
  --batch-size 1024
