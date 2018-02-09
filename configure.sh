#!/bin/bash
BASE=$PWD

## Initialise submodules
git submodule update --init --recursive


## Install pycocotools
cd $BASE/third-party/cocoapi/PythonAPI
if [[ $1 == python3 ]] ; then
 echo "Building pycocotools for python3"
 sed -r -i 's/python\s/python3 /g' Makefile
else
 echo "Building pycocotools for python2"
fi
make && sudo make install

## Download Mask RCNN pretrained weights and patch coco file
cd $BASE/third-party/Mask_RCNN
git apply --stat ../../maskrcnn-coco.patch
[[ ! -e mask_rcnn_coco.h5 ]] && \
 wget -c https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5


## Download Mask RCNN tutorial datasets
# cd $BASE/datasets
# [[ ! -e instances_minival2014.json ]] && \
#  wget -c https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip && \
#  unzip instances_minival2014.json.zip 
# [[ ! -e instances_valminusminival2014.json ]] && \
#  wget -c https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip &&
#  unzip instances_valminusminival2014.json.zip

## Set MASK RCNN path for python scripts
[[ ! $(grep 'export MASK_RCNN_PATH' ~/.profile) ]] && \
 echo "export MASK_RCNN_PATH=$BASE/third-party/Mask_RCNN" >> ~/.profile

