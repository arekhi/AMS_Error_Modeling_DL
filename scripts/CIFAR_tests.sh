#!/bin/bash

python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /arekhi.scratch.DL/checkpoints/resnet-50 \
  --arch resnet44_cifar \
  --pretrained \
  --evaluate
#  --Dorefa 8 8 \
#  --workers 24 \
#  --epochs 200
