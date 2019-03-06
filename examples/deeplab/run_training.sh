#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class
CHECKPOINT=/home/users/spo/src/tensorflow_models/research/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt
DEEPLAB_DIR=/home/users/spo/src/tensorflow_models/research/deeplab

# Without BN
CUDA_VISIBLE_DEVICES=0 \
#bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
# --deeplab-dir $DEEPLAB_DIR \
# --checkpoint $CHECKPOINT \
# --learning-rate 1e-4 \
# --train-steps 20000 \
# --reinitialize-head \
#--train-freeze-batch-norm \
# train \
# $LOGDIR/tfrecord \

# With BN
CUDA_VISIBLE_DEVICES=0 \
bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
 --deeplab-dir $DEEPLAB_DIR \
 --checkpoint $CHECKPOINT \
 --learning-rate 1e-4 \
 --train-steps 20000 \
 --reinitialize-head \
 train \
 $LOGDIR/tfrecord \
 $LOGDIR
