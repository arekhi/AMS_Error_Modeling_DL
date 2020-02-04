#!/bin/bash

python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /arekhi.scratch.DL/checkpoints/resnet-50/precision_sweep \
  --arch resnet50 \
  --pretrained \
  --workers 30 \
  --epochs 200 \
  --batch-size 1024 \
  --compress /arekhi.scratch.DL/scripts/resnet50_imagenet_dorefa.yaml \
  --lr 4e-3
