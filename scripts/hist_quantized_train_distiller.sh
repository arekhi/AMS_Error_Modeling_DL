#!/bin/bash

python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
  /projects.cdr/ImageNet \
  --out-dir /arekhi.scratch.DL/checkpoints/resnet-50/hyperparam \
  --arch resnet50 \
  --pretrained \
  --workers 24 \
  --epochs 100 \
  --batch-size 256 \
  --compress /arekhi.scratch.DL/scripts/resnet50_imagenet_dorefa.yaml \
  --lr 0.001 \
  --param-hist
