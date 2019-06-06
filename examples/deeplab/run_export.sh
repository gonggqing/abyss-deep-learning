#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190228.2_class_noBN
DEEPLAB_DIR=/home/users/spo/src/tensorflow_models/research/deeplab

EXPORT_CROP_SIZE=4097

CUDA_VISIBLE_DEVICES=0,0 \
bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
 --deeplab-dir $DEEPLAB_DIR \
 --export-checkpoint-number 17673 \
 --export-crop-size $EXPORT_CROP_SIZE \
 export \
 $LOGDIR/tfrecord \
 $LOGDIR
