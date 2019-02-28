#!/bin/bash
set -e

[[ $# -lt 1 ]] && { echo "Usage: ./configure [python3 | python2]"; exit 1; }

BASE=$PWD
SRC_DIR=~/src
COCOTOOLS_DIR=$SRC_DIR/cocoapi

# ## Initialise submodules
# git submodule update --init --recursive

if [[ ! -e $SRC_DIR ]] ; then
	mkdir "$SRC_DIR"
fi
if [[ ! -e $COCOTOOLS_DIR ]] ; then
  cd "$SRC_DIR"
  git clone https://github.com/cocodataset/cocoapi.git
fi

## Install pycocotools
cd "$COCOTOOLS_DIR/PythonAPI"
if [[ $1 == python3 ]] ; then
 echo "Building pycocotools for python3"
 sed -r -i 's/python\s/python3 /g' Makefile
else
 echo "Building pycocotools for python2"
fi
make && sudo make install

## Download Mask RCNN pretrained weights and update remotes
#mkdir -p "$BASE/abyss_maskrcnn"
#cd "$BASE/abyss_maskrcnn"
#[[ ! -e mask_rcnn_coco.h5 ]] && \
# wget -c https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

## Download Mask RCNN tutorial datasets
# cd $BASE/datasets
# [[ ! -e instances_minival2014.json ]] && \
#  wget -c https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip && \
#  unzip instances_minival2014.json.zip 
# [[ ! -e instances_valminusminival2014.json ]] && \
#  wget -c https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip &&
#  unzip instances_valminusminival2014.json.zip
