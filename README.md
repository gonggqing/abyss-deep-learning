deep-learning
=================
## Contents
   * [deep-learning](#deep-learning)
      * [Introduction](#introduction)
      * [Installation](#installation)
         * [Local](#local)
         * [Docker](#docker)
      * [Applications](#applications)
      * [Example: BAE Prop data with labelme labels](#example-bae-prop-data-with-labelme-labels)
         * [Important notes](#important-notes)
         * [Prerequisites](#prerequisites)
         * [Overview](#overview)
         * [Commands](#commands)
      * [coco-to-csv: Convert COCO JSON into CSV](#coco-to-csv-convert-coco-json-into-csv)
         * [Example:](#example)

## Introduction
Provides tools to manipulate COCO JSON, VOC CSV datasets as well as Mask RCNN train-val, test and predict.

As of April 2018 the MASK_RCNN_PATH environment variable is no longer needed as the distutils repo has been merged to master.

## Examples and demos
See the [wiki](https://github.com/abyss-solutions/deep-learning/wiki) for examples and demos.

## Installation
### Local
```bash 
mkdir -p ~/src/abyss
cd ~/src/abyss
git clone https://github.com/abyss-solutions/deep-learning.git
cd deep-learning
./configure.sh
```

### Docker
First initialise the repo:
```bash 
mkdir -p ~/src/abyss
cd ~/src/abyss
git clone https://github.com/abyss-solutions/deep-learning.git
cd deep-learning/docker
chmod 600 ssh/*
docker build -t abyss/dl .
```

Add the following alias to your host that will allow you to run the image easily:
```bash
echo 'alias docker-dl="nvidia-docker run --user docker -it --rm -v /home/$USER:/home/docker -v /:/host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 8888:8888 -p 7001:7001 abyss/dl bash"' > ~/.abyss_aliases
source ~/.abyss_aliases
```
 
Now run docker with xhost sharing:
```bash
xhost +local:root
xhost +local:$USER
docker-dl
```
Alternative:
```bash
echo 'alias docker-dl="nvidia-docker run --user docker --net=host -e DISPLAY=$DISPLAY -it -p 8888:8888 -p 7001:7001 -p 6006:6006 --volume "$HOME/.Xauthority:/root/.Xauthority:rw" -v ~/src:/home/docker/src -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /:/host --rm abyss/dl bash"' > ~/.abyss_aliases
source ~/.abyss_aliases
docker-dl
```

TODO: X forwarding from docker>local when ssh'ed in to hippo.

You should now be in an environment that will have all the prerequisites for you to install the deep-learning repo:
```bash
cd ~/src/abyss/deep-learning
sudo ./configure.sh python3 # This will take a while to download the MaskRCNN weights
sudo python3 setup.py install
```

Now you are ready to run maskrcnn-trainval, etc...

To run Jupyter Notebook from Docker whilst ssh'ed to hippo:
```bash
jupyter notebook --no-browser --port 9000
```

Then in a new local terminal:
```bash
ssh -N -f -L localhost:8889:0.0.0.0:9000 jmc@hippo
```
Now visit localhost:8889/?token= _____ with the token given by the jupyter server.

## Applications
* coco-calc-masks: Open a COCO dataset and save a new one, where the segmentations are always masks.
* coco-check-data-pollution: Examine combinations of COCO datasets to ensure there are no common images between them.
* coco-extract-masks: Extract masks from COCO to pngs
* coco-merge: Merge multiple COCO datasets
* coco-split: Split a COCO dataset into subsets given split ratios
* coco-to-csv: Convert a COCO dataset into a series of CSV files (similar to VOC format)
* coco-from-video: For one or more labeled videos, create COCO json files and, optionally, corresponding image frames
* coco-viewer: View coco images and annotations, calculate RGB mean pixel
* image-dirs-to-coco: Convert VOC style annotations into COCO
* labelme-to-coco: Convert labelme dataset into COCO
* maskrcnn-find-lr: Search in log space for a suitable learning rate for a network and dataset.
* maskrcnn-predict: Predict on input images or streams using a trained Mask RCNN model and output labels or overlay
* maskrcnn-test: Test a trained network on a COCO Dataset
* maskrcnn-trainval: Train a Mask RCNN network with COCO dataset* 

## Repo Summary - WIP
* abyss_deep_learning
  * base
  * datasets
  * keras
  * abyss_dataset.py - 
  * coco_classes.py
  * metrics.py
  * ocr.py
  * utils.py
    * cv2_to_Pil()
    * instance_to_caption() - deprecated, replaced by Translator classes
    * config_gpu()
    * import_config() - generic json/yaml config reader
    * balanced_set()
    * tile_gen()
    * detile()
    * instance_to_categorical()
    * cat_to_onehot()
    * ann_rle_encode()
    * warn_on_call()
    * warn_once()
    * image_streamer()
  * visualize.py
* applications
* configs
* data
* docker
* jupyter
