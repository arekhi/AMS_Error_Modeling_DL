#!/bin/bash

python /arekhi.scratch.DL/models/research/slim/eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=/arekhi.scratch.DL/checkpoints/vgg_16.ckpt \
  --checkpoint_exclude_scopes=vgg_16/test/biases,vgg_16/test/weights \
  --dataset_dir=/projects.cdr/ImageNet/train-val-tfrecord/ \
  --dataset_name=imagenet \
  --dataset_split_name=validation \
  --model_name=vgg_16 \
  --labels_offset=1 \
  --eval_dir=/arekhi.scratch.DL/tmp/tensorboard/vgg16/
