#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class_noBN
DEEPLAB_DIR=/home/users/spo/src/tensorflow_models/research/deeplab

CUDA_VISIBLE_DEVICES=0,0 \
bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
 --deeplab-dir $DEEPLAB_DIR \
 eval \
 $LOGDIR/tfrecord \
 $LOGDIR
