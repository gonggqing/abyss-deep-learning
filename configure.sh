#!/bin/bash

BASE=$PWD
cd $BASE/third-party/cocoapi/PythonAPI
# sed -r -i 's/python\s/python3 /g' Makefile
make && sudo make install

cd $BASE/third-party/Mask_RCNN
[[ ! -e mask_rcnn_coco.h5 ]] && \
 wget -c https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

cd $BASE/datasets

#[[ ! -e instances_minival2014.json ]] && \
# wget -c https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip && \
# unzip instances_minival2014.json.zip 
#[[ ! -e instances_valminusminival2014.json ]] && \
# wget -c https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip &&
# unzip instances_valminusminival2014.json.zip

