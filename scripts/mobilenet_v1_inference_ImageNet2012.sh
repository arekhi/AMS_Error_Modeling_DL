#!/bin/bash

python -m pdb /arekhi.scratch.DL/models/research/slim/mobilenet_v1_eval.py \
  --alsologtostderr \
  --checkpoint_dir=/arekhi.scratch.DL/checkpoints/mobilenet_v1/mobilenet_v1_1.0_224_quant.ckpt \
  --dataset_dir=/projects.cdr/ImageNet/train-val-tfrecord/ \
  --eval_dir=/arekhi.scratch.DL/tmp/tensorboard/mobilenet_v1/ \
  --labels_offset=1 \
  --quantize=True
