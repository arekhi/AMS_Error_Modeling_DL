#!/bin/bash

python /arekhi.scratch.DL/models/research/slim/eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=/arekhi.scratch.DL/checkpoints/inception_resnet_v2_2016_08_30.ckpt \
  --dataset_dir=/projects.cdr/ImageNet/train-val-tfrecord/ \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=inception_resnet_v2
