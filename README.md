# deep-learning

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
cd docker
docker build -t abyss/dl .
```

Add the following alias to your host that will allow you to run the image easily:
```bash
echo 'alias docker-dl="nvidia-docker run --user docker -it --rm -v /home/$USER:/home/docker -v /:/host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 8888:8888 -p 7001:7001 abyss/dl bash"' > ~/.abyss_aliases
source ~/.bash_aliases
```

Now run docker with xhost sharing:
```bash
xhost +local:root
xhost +local:$USER
docker-dl
```

You should now be in an environment that will allow you to run any of the below applications.

## Applications
* coco-calc-masks: Open a COCO dataset and save a new one, where the segmentations are always masks.
* coco-check-data-pollution: Examine combinations of COCO datasets to ensure there are no common images between them.
* coco-extract-masks: Extract masks from COCO to pngs
* coco-merge: Merge multiple COCO datasets
* coco-split: Split a COCO dataset into subsets given split ratios
* coco-to-csv: Convert a COCO dataset into a series of CSV files (similar to VOC format)
* coco-viewer: View coco images and annotations, calculate RGB mean pixel
* image-dirs-to-coco: Convert VOC style annotations into COCO
* labelme-to-coco: Convert labelme dataset into COCO
* maskrcnn-find-lr: Search in log space for a suitable learning rate for a network and dataset.
* maskrcnn-predict: Predict on input images or streams using a trained Mask RCNN model and output labels or overlay
* maskrcnn-test: Test a trained network on a COCO Dataset
* maskrcnn-trainval: Train a Mask RCNN network with COCO dataset* 

## Example: BAE Prop data with labelme labels
### Important notes
*  skimage.io.imread sometimes does not read the image properly if it is not PNG; typically this happens if JPGs are large.
    It's highly suggested to convert all images to PNGs before starting:
    `bash for i in *.jpg; do convert $i ${i:0:-4}.png; done`
*  There are changes you will need to pull from steve's fork of the original Mask_RCNN code:
   `bash git remote add steve https://github.com/spotiris/Mask_RCNN.git`
   `bash git pull steve master`

### Prerequisites
* Download the labelme collection and extract it
* Have this repo downloaded and the correct env vars set up (should be done when you ./configure.sh)

### Overview
This example will show you how to:
* Use labelme-to-coco to convert the dataset
* Use coco-split to split the dataset into train and val datasets with 75% of the images in train and 25% of the images in val
* Use maskrcnn-trainval to train the model
* Use maskrcnn-predict to output predictions

Directory structure downloaded from labelme collection:
```
/data/abyss/bae-prop-uw
.
├── Annotations
│   └── users
│       └── sbargoti
│           └── baeprop <-- xml annotations here
├── Images
│   └── users
│       └── sbargoti
│           └── baeprop  <-- rgb images here
├── Masks
│   └── users
│       └── sbargoti
│           └── baeprop
└── Scribbles
    └── users
        └── sbargoti
            └── baeprop
```

### Commands
Prepare dataset:
```bash
~/src/abyss/deep-learning/applications/labelme-to-coco \
    /data/abyss/bae-prop-uw/Annotations/users/sbargoti/baeprop > /data/abyss/bae-prop-uw/baeprop-coco.json
~/src/abyss/deep-learning/applications/coco-split \
    /data/abyss/bae-prop-uw/baeprop-coco.json train,val 0.75,0.25 \
    --image-dir /data/abyss/bae-prop-uw/Images/users/sbargoti/baeprop
```

Train:
We will train the classification head, then the whole network (feature detector + head).
This should give a better result than just the head and be faster and more stable than just training 'all'.

Note: Mask RCNN implementation does something where it will reset the heads layer if it does not recognise that logdir and filename of the weights you provide.
It is recommended to use the `--weights 'last'` way of training the 'all' layers after the 'heads' layers to ensure it picks up the right one.

```bash
~/src/abyss/deep-learning/applications/maskrcnn-trainval \
    /data/abyss/bae-prop-uw/baeprop-coco_train.json \
    /data/abyss/bae-prop-uw/baeprop-coco_val.json \
    $MASK_RCNN_PATH/logs \
    --config $MASK_RCNN_PATH/../../configs/MaskRCNN_default_config.py \
    --weights $MASK_RCNN_PATH/mask_rcnn_coco.h5 \
    --image-dir /data/abyss/bae-prop-uw/Images/users/sbargoti/baeprop \
    --epochs 1 --layers heads

~/src/abyss/deep-learning/applications/maskrcnn-trainval \
    /data/abyss/bae-prop-uw/baeprop-coco_train.json \
    /data/abyss/bae-prop-uw/baeprop-coco_val.json \
    $MASK_RCNN_PATH/logs \
    --config $MASK_RCNN_PATH/../../configs/MaskRCNN_default_config.py \
    --weights last \
    --image-dir /data/abyss/bae-prop-uw/Images/users/sbargoti/baeprop \
    --epochs 1 --layers all
```
Watch the training, see the error go down, and find the new model that has been written to $MASK_RCNN_PATH/logs/default**/mask_rcnn_default-model_0001.h5 (it will be in the most recent directory created).

Predict:
```bash
~/src/abyss/deep-learning/applications/maskrcnn-predict \
    $MASK_RCNN_PATH/../../configs/MaskRCNN_default_config.py \
    $MASK_RCNN_PATH/logs/default-model20180209T0337/mask_rcnn_default-model_0001.h5 \
    $MASK_RCNN_PATH/logs \
    /data/abyss/bae-prop-uw/Images/users/sbargoti/baeprop/*.png \
    --show
```
If you want to save the images remove --show and add any of: --rgb-labels, --overlay.

Note this is a bad dataset - results are not that good.
Network may be better with other configs.

## coco-to-csv: Convert COCO JSON into CSV

This utility is to convert to CSV for Suchet's Faster RCNN.
It by default outputs id,bbox,category_id but can be overridden with --fields flag.
The output format for bbox is bbox_x,bbox_y,bbox_w,bbox_h.

Output fields that can be used:
* id: The unique ID of the annotation
* bbox: The bounding box of the annotation (x, y, w, h)
* category_id: The ID of the category of the annotation
* area: The area of the annotation (calcualted from mask/poly or bbox, in that order)
* iscrowd: 1 if the annotation is a group of objects.


### Example: 
```bash
cd ~/src/abyss/deep-learning/applications
./coco-to-csv \
   /data/abyss/bae-prop-uw/baeprop-coco.json \
   /data/abyss/bae-prop-uw/annotations-csv \
   --fields id,bbox,category_id --verbose
```
