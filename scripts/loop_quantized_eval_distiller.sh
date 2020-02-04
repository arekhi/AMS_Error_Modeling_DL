#!/bin/bash

for epoch in {125..90..5}
do
  resumePath="/arekhi.scratch.DL/checkpoints/resnet-50/precision_sweep/2018.09.28-001748/epoch_${epoch}_checkpoint.pth.tar"
  python3 /arekhi.scratch.DL/distiller/examples/classifier_compression/compress_classifier.py \
    /projects.cdr/ImageNet \
    --out-dir /arekhi.scratch.DL/checkpoints/resnet-50/tmp \
    --arch resnet50 \
    --resume "${resumePath}" \
    --evaluate \
    --workers 30 \
    --batch-size 1024
done
