#!/bin/bash

python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /arekhi.scratch.DL/checkpoints/resnet-50/precision_sweep \
  --arch resnet50 \
  --resume /arekhi.scratch.DL/checkpoints/resnet-50/precision_sweep/2018.09.23-051546/checkpoint.pth.tar \
  --workers 30 \
  --batch-size 1024 \
  --lr 4e-3
