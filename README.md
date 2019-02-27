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

## Examples and demos
See the [wiki](https://github.com/abyss-solutions/deep-learning/wiki) for examples and demos.

## Installation
### Local
```bash 
mkdir -p ~/src/abyss
cd ~/src/abyss
git clone https://github.com/abyss-solutions/deep-learning.git
cd deep-learning
pip3 install --user .
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
sudo ./setup-host.sh
```

Then run the docker container and do the post-install setup using:
```bash
source ~/.abyss-aliases.sh
abyss-dl
# Now you will be in the docker container
~/post-install.sh # This builds and installs crfasrnn_keras, needs to be done separately for now
```
You should now be in an environment that will have all the prerequisites installed. 

#### Structure
All files in docker are non-persistent, with the exception of these special directories:
* /scratch: A place to write your project files
* /data: Typically read-only data from datasets

#### git repos
The git repos in deep-learning docker are all equipped with deployment keys, meaning the repos do not require a user to login to clone or pull.
You cannot push to the repo using the deployment keys.
In order to pull, use the script in `/home/docker/bin/git-deployment-pull repo_name`.

#### Versions
In the docker installer the following versions are used:
* Keras 2.2.2
* Tensorflow (GPU) 1.9.0
* CUDA 9.0

## Running Jupyter Notebook
In your docker container run:
```bash
jupyter notebook
```
Now visit localhost:8888/?token=_____ with the token given by the jupyter server.
The token must be used the first time, then you can use the password "123".

## Applications
* coco-calc-masks: Open a COCO dataset and save a new one, where the segmentations are always masks.
* coco-check-data-pollution: Examine combinations of COCO datasets to ensure there are no common images between them.
* coco-extract-masks: Extract masks from COCO to pngs
* coco-draw: dump coco annotations in the form of images with overlayed detections. Supports coco segmentation and bounding boxes
* coco-from-video: For one or more labeled videos, create COCO json files and, optionally, corresponding image frames 
* coco-grep: For given json files, search for all images of a specific caption type
* coco-merge: For given json files, merge them into a single json file
* coco-repath: TODO
* coco-sample: For a give json file, randomly sample N images from all images
* coco-split: Split a COCO dataset into subsets given split ratios
* coco-subsample-balanced:
* coco-to-csv: Convert a COCO dataset into a series of CSV files (similar to VOC format)
* coco-to-yolo3: Convert a COCO dataset into a YOLO3 dataset (https://github.com/qqwweee/keras-yolo3/tree/master/yolo3)
* coco-viewer: View coco images and annotations, calculate RGB mean pixel
* image-dirs-to-coco: Convert VOC style annotations into COCO
* keras-graph: TODO
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
