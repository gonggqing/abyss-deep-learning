# deep-learning

## Installation
```bash 
mkdir -p ~/src/abyss
cd ~/src/abyss
git clone https://github.com/abyss-solutions/deep-learning.git
cd deep-learning
./configure.sh
```

## Applications
* labelme-to-coco: Convert labelme dataset into COCO JSON
* coco-extract-masks: Extract masks from COCO JSON to pngs
* coco-viewer: View coco images and annotations
* maskrcnn-trainval: Train a Mask RCNN network with COCO JSON dataset
* maskrcnn-predict: Predict on input images or streams using a trained Mask RCNN model and output labels or overlay

## TODO
* maskrcnn-test: Test a Mask RCNN network with COCO JSON dataset
* maskrcnn-predict
  * Add feature: Take in STDIN cv-cat stream and output prediction cv-cat stream
* Tutorials

## Example: BAE Prop data with labelme labels
### Important note
skimage.io.imread sometimes does not read the image properly if it is not PNG; typically this happens if JPGs are large.
It's highly suggested to convert all images to PNGs before starting:

`bash for i in *.jpg; do convert $i ${i:0:-4}.png; done`

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
This should give a better result than just the head.
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
