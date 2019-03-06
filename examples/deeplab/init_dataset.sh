#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class
mkdir -p $LOGDIR

~/src/abyss/deep-learning/applications/deeplabv3+ \
  --deeplab-dir /home/users/spo/src/tensorflow_models/research/deeplab \
  --image-dir "/mnt/rhino/processed/industry-data/anadarko/gunnison/r2s/sphericals/CD -/point-cuts" \
  --eval-dataset /mnt/rhino/processed/industry-data/anadarko/gunnison/labels/point-cuts/2nd_run/all_2class.val.json \
  --image-format png \
  init \
  /mnt/rhino/processed/industry-data/anadarko/gunnison/labels/point-cuts/2nd_run/all_2class.train.json \
  $LOGDIR
